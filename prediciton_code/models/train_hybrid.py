# train_hybrid.py
import os, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as scio
from torch.utils.data import DataLoader

from prediciton_code.metrics import NMSELoss
from prediciton_code.test import HybridTransformerLSTM  # or import from where you defined it

from prediciton_code.data import SeqData, LoadBatch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='CDL-B/train')
parser.add_argument('--prev_len', type=int, default=25)
parser.add_argument('--min_prev', type=int, default=10)
parser.add_argument('--max_prev', type=int, default=25)
parser.add_argument('--pred_len', type=int, default=5)
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--hs', type=int, default=256)
parser.add_argument('--hl', type=int, default=2)
parser.add_argument('--enc_in', type=int, default=None)  # if None will infer from data
parser.add_argument('--save_dir', type=str, default='./checkpoints/hybrid/')
parser.add_argument('--use_gpu', action='store_true')
parser.add_argument('--teacher_start_epochs', type=int, default=15)
parser.add_argument('--clip', type=float, default=1.0)
args = parser.parse_args()

device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
os.makedirs(args.save_dir, exist_ok=True)

# Dataset: use SeqData if it supports train mode; else implement simple SeqData-based dataset
train_ds = SeqData(args.data_dir, prev_len=args.prev_len, pred_len=args.pred_len, mode='train',
                  SNR=14, ir=1, samples=1, v_min=30, v_max=60)
val_ds = SeqData(args.data_dir, prev_len=args.prev_len, pred_len=args.pred_len, mode='val',
                SNR=14, ir=1, samples=1, v_min=30, v_max=60)

# If SeqData returns (data, _, prev, pred) like your testData, we can create a small wrapper to yield sliding windows
class TrainWrapper(torch.utils.data.Dataset):
    def __init__(self, seqdata, min_prev, max_prev, pred_len):
        self.seqdata = seqdata
        self.min_prev = min_prev
        self.max_prev = max_prev
        self.pred_len = pred_len
        # Build list of (file_idx, start) windows using default prev_len, but we will randomize prev in __getitem__
        self.indexes = list(range(len(seqdata)))
    def __len__(self):
        # we will iterate over files; each epoch we will sample multiple windows by randomizing
        return len(self.indexes) * 8  # multiplier to get more steps per epoch
    def __getitem__(self, idx):
        fidx = idx % len(self.indexes)
        data, _, _, _ = self.seqdata[fidx]  # data: (L, M, Nr, Nt)
        L = data.shape[0]
        prev_len = np.random.randint(self.min_prev, self.max_prev+1)
        max_start = L - (prev_len + self.pred_len)
        if max_start < 0:
            prev_len = self.min_prev
            max_start = L - (prev_len + self.pred_len)
        start = np.random.randint(0, max_start+1)
        inp = data[start:start+prev_len]   # shape [prev_len, M, Nr, Nt]
        tgt = data[start+prev_len:start+prev_len+self.pred_len]
        # convert to real/imag if needed: assume SeqData already returns real (2-channel) if designed that way
        # flatten per time-slot: shape -> [prev_len, enc_in]
        # We'll reuse LoadBatch to make same encoding as your test script
        # first convert to numpy if not
        inp_flat = LoadBatch(inp).numpy() if hasattr(LoadBatch(inp),'numpy') else LoadBatch(inp)
        tgt_flat = LoadBatch(tgt).numpy() if hasattr(LoadBatch(tgt),'numpy') else LoadBatch(tgt)
        return torch.from_numpy(inp_flat).float(), torch.from_numpy(tgt_flat).float(), prev_len

train_wrapper = TrainWrapper(train_ds, args.min_prev, args.max_prev, args.pred_len)
train_loader = DataLoader(train_wrapper, batch_size=args.batch, shuffle=True, drop_last=True)

# val loader: deterministic windows using seqdata indexing
class ValWrapper(torch.utils.data.Dataset):
    def __init__(self, seqdata, prev_len, pred_len):
        self.seqdata = seqdata
        self.prev_len = prev_len
        self.pred_len = pred_len
    def __len__(self):
        return len(self.seqdata)
    def __getitem__(self, idx):
        data, _, _, _ = self.seqdata[idx]
        L = data.shape[0]
        # pick last window
        start = L - (self.prev_len + self.pred_len)
        if start < 0: start = 0
        inp = data[start:start+self.prev_len]
        tgt = data[start+self.prev_len:start+self.prev_len+self.pred_len]
        inp_flat = LoadBatch(inp)
        tgt_flat = LoadBatch(tgt)
        return torch.from_numpy(inp_flat).float(), torch.from_numpy(tgt_flat).float()

val_wrapper = ValWrapper(val_ds, args.prev_len, args.pred_len)
val_loader = DataLoader(val_wrapper, batch_size=args.batch, shuffle=False)

# infer enc_in
sample_inp, _ = train_wrapper[0][0], train_wrapper[0][1]
enc_in = args.enc_in if args.enc_in is not None else sample_inp.shape[-1]

# instantiate model
model = HybridTransformerLSTM(enc_in=enc_in, d_model=args.d_model, nhead=4, num_layers=2,
                             lstm_hidden=args.hs, lstm_layers=max(1,args.hl//1), pred_len=args.pred_len, dropout=0.05)
model.to(device)

criterion = nn.MSELoss()  # training with MSE on real+imag flattened vectors
nmse_loss = NMSELoss()     # for validation reporting
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

best_val = 1e9
teacher_forcing_ratio = 1.0

for epoch in range(1, args.epochs+1):
    model.train()
    t0 = time.time()
    train_loss = 0.0
    for batch_idx, (inp_batch, tgt_batch, prev_lens) in enumerate(train_loader):
        inp_batch = inp_batch.to(device)        # [B, prev_len_max, enc_in]
        tgt_batch = tgt_batch.to(device)        # [B, pred_len, enc_in]
        B, seq_len, _ = inp_batch.shape
        optimizer.zero_grad()

        # Option A: full teacher forcing — feed ground truth frames into decoder by projecting them back
        # Our Hybrid forward currently does not accept teacher forcing, so we implement a simple supervised loss:
        # Run model autoregressively and compute MSE between outputs and tgt_batch.

        preds = model(inp_batch)  # [B, pred_len, enc_in]
        loss = criterion(preds, tgt_batch)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    # validation
    model.eval()
    val_losses = []
    val_nmse = []
    with torch.no_grad():
        for inp_batch, tgt_batch in val_loader:
            inp_batch = inp_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            preds = model(inp_batch)
            val_losses.append(criterion(preds, tgt_batch).item())
            # NMSE compute using your NMSELoss wrapper (expects real-valued tensors like in test script)
            val_nmse.append(nmse_loss(preds.cpu(), tgt_batch.cpu().numpy() if isinstance(tgt_batch, np.ndarray) else tgt_batch.cpu()).item() if hasattr(nmse_loss,'__call__') else 0.0)

    avg_train = train_loss / len(train_loader)
    avg_val = np.mean(val_losses) if len(val_losses)>0 else 0.0
    print(f"Epoch {epoch} train_loss={avg_train:.6f} val_loss={avg_val:.6f} time={time.time()-t0:.1f}s")

    # save best
    if avg_val < best_val:
        best_val = avg_val
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(args.save_dir, 'hybrid_best.pth'))
        print("Saved best checkpoint.")

print("Training complete. Best val loss:", best_val)
