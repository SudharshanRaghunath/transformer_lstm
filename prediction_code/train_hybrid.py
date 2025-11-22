#!/usr/bin/env python3
"""
train_hybrid.py

Train HybridTransformerLSTM on your CDL-B dataset.

Usage example:
python train_hybrid.py --data_dir CDL-B/train --save_dir ./checkpoints/hybrid --epochs 20 --batch 8 --use_gpu 0
"""
import os
import time
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.io as scio
import matplotlib.pyplot as plt

# repo imports
from data import SeqData, LoadBatch
from metrics import NMSELoss
from utils import real2complex
from models.hybrid import HybridTransformerLSTM

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='CDL-B/train')
    p.add_argument('--save_dir', type=str, default='./checkpoints/hybrid')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--use_gpu', type=int, default=0)
    p.add_argument('--gpu_list', type=str, default='0')

    p.add_argument('--seq_len', type=int, default=25)
    p.add_argument('--pred_len', type=int, default=5)
    p.add_argument('--enc_in', type=int, default=16)
    p.add_argument('--d_model', type=int, default=64)
    p.add_argument('--hs', type=int, default=256)
    p.add_argument('--hl', type=int, default=2)

    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-6)
    p.add_argument('--tf_warmup', type=int, default=10, help='epochs with TF=1.0')
    p.add_argument('--final_tf', type=float, default=0.0)

    p.add_argument('--v_min', type=int, default=30)
    p.add_argument('--v_max', type=int, default=60)
    p.add_argument('--ir', type=int, default=1)
    p.add_argument('--samples', type=int, default=1)
    p.add_argument('--SNR', type=float, default=14.0)
    return p.parse_args()


def combined_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.6):
    mse = nn.MSELoss()(pred, target)
    # nmse using flattened tensors
    num = torch.sum((pred - target).abs() ** 2, dim=(1, 2))
    den = torch.sum((target).abs() ** 2, dim=(1, 2)) + 1e-12
    nmse = torch.mean(num / den)
    return alpha * nmse + (1.0 - alpha) * mse


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_checkpoint(model, optimizer, epoch, path):
    payload = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(payload, path)


def _flatten_batch_if_needed(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts x possibly of shape:
      - [B, seq_len, feat]  (already good)
      - [batch_outer, M, seq_len, feat] (common with your SeqData+LoadBatch)
    Returns a tensor shaped [B_total, seq_len, feat] where B_total = batch_outer * M.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if x.dim() == 3:
        return x
    if x.dim() == 4:
        b_out, M, seq_len, feat = x.shape
        return x.reshape(b_out * M, seq_len, feat)
    # if shape unexpected, try to coerce
    return x.view(-1, x.shape[-2], x.shape[-1])  # best-effort


def train_one_epoch(model, optimizer, loader, device, args, epoch):
    model.train()
    losses = []
    for batch_idx, batch in enumerate(loader):
        # batch is tuple returned by Dataset wrapper: (inp_flat, fut_flat)
        inp_flat, fut_flat = batch

        # If DataLoader collated numpy arrays into tensors on CPU, convert to torch
        if isinstance(inp_flat, np.ndarray):
            inp_flat = torch.from_numpy(inp_flat).float()
        if isinstance(fut_flat, np.ndarray):
            fut_flat = torch.from_numpy(fut_flat).float()

        # Move to device if not already
        inp_flat = inp_flat.to(device)
        fut_flat = fut_flat.to(device)

        # Flatten batch dims if dataset returned [batch_outer, M, seq_len, feat]
        inp_flat = _flatten_batch_if_needed(inp_flat)
        fut_flat = _flatten_batch_if_needed(fut_flat)

        optimizer.zero_grad()

        # teacher forcing schedule
        try:
            tf_ratio = 1.0 if epoch <= args.tf_warmup else max(
                args.final_tf,
                1.0 - (epoch - args.tf_warmup) / float(max(1, args.epochs - args.tf_warmup))
            )
            # model.forward supports teacher_forcing_ratio and optional tgt
            preds = model(inp_flat, tgt=fut_flat, teacher_forcing_ratio=tf_ratio)
        except TypeError:
            preds = model.generate(inp_flat, pred_len=args.pred_len)

        loss = combined_loss(preds, fut_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


def validate(model, loader, device, args):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            inp_flat, fut_flat = batch
            if isinstance(inp_flat, np.ndarray):
                inp_flat = torch.from_numpy(inp_flat).float()
            if isinstance(fut_flat, np.ndarray):
                fut_flat = torch.from_numpy(fut_flat).float()
            inp_flat = inp_flat.to(device)
            fut_flat = fut_flat.to(device)

            inp_flat = _flatten_batch_if_needed(inp_flat)
            fut_flat = _flatten_batch_if_needed(fut_flat)

            try:
                preds = model.generate(inp_flat, pred_len=args.pred_len)
            except Exception:
                preds = model(inp_flat, tgt=fut_flat, teacher_forcing_ratio=0.0)

            loss = combined_loss(preds, fut_flat)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def make_wrap_dataset(seqdata):
    class Wrapper(torch.utils.data.Dataset):
        def __init__(self, seqdata):
            self.seqdata = seqdata

        def __len__(self):
            return len(self.seqdata)

        def __getitem__(self, idx):
            data, _, prev, fut = self.seqdata[idx]
            inp_flat = LoadBatch(prev)
            fut_flat = LoadBatch(fut)
            if isinstance(inp_flat, np.ndarray):
                inp_flat = torch.from_numpy(inp_flat).float()
            if isinstance(fut_flat, np.ndarray):
                fut_flat = torch.from_numpy(fut_flat).float()
            # ensure shapes: [M, seq_len, enc_in] — treat M as batch dimension
            # Many of your SeqData objects return per-sample M items as a batch; that's expected
            return inp_flat, fut_flat

    return Wrapper(seqdata)


def main():
    args = parse_args()
    ensure_dir(args.save_dir)
    print("Train: saving to", args.save_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
    device = torch.device('cuda:0') if (args.use_gpu and torch.cuda.is_available()) else torch.device('cpu')
    print("Using device:", device)

    # load datasets
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"data_dir not found: {args.data_dir}")
    train_ds = SeqData(args.data_dir, prev_len=args.seq_len, pred_len=args.pred_len,
                       mode='train', SNR=args.SNR, ir=args.ir, samples=args.samples,
                       v_min=args.v_min, v_max=args.v_max)
    # optional val - if missing, use a small split from train (SeqData should support mode='val')
    try:
        val_ds = SeqData(args.data_dir, prev_len=args.seq_len, pred_len=args.pred_len,
                         mode='val', SNR=args.SNR, ir=args.ir, samples=args.samples,
                         v_min=args.v_min, v_max=args.v_max)
        has_val = len(val_ds) > 0
    except Exception:
        val_ds = None
        has_val = False

    train_wrap = make_wrap_dataset(train_ds)
    train_loader = DataLoader(train_wrap, batch_size=args.batch, shuffle=True, drop_last=True)

    if has_val and val_ds:
        val_wrap = make_wrap_dataset(val_ds)
        val_loader = DataLoader(val_wrap, batch_size=args.batch, shuffle=False)
    else:
        val_loader = None

    # model
    model = HybridTransformerLSTM(enc_in=args.enc_in, d_model=args.d_model, nhead=4,
                                  trans_layers=2, lstm_hidden=args.hs, lstm_layers=max(1, args.hl),
                                  pred_len=args.pred_len, dropout=0.05)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    best_val = 1e9
    criterion = NMSELoss()

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, args, epoch)
        if val_loader:
            val_loss = validate(model, val_loader, device, args)
        else:
            val_loss = train_loss  # fallback if no val set

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(
            f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} time={time.time() - t0:.1f}s")

        # save checkpoint every epoch (debug + reproducibility)
        epoch_ckpt = os.path.join(args.save_dir, f"hybrid_epoch{epoch}.pth")
        try:
            save_checkpoint(model, optimizer, epoch, epoch_ckpt)
            print("Saved epoch checkpoint:", epoch_ckpt)
        except Exception as e:
            print("Error saving epoch checkpoint:", e)

        # save best by validation loss (NMSE)
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.save_dir, "hybrid_best.pth")
            try:
                save_checkpoint(model, optimizer, epoch, best_path)
                print("Saved best checkpoint:", best_path)
            except Exception as e:
                print("Error saving best checkpoint:", e)

    # final plots and save history
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(args.save_dir, 'training_loss.png'))
    plt.close()

    print("Training finished. Best val:", best_val)
    # save a small sample prediction (first sample from train) for inspection
    try:
        data, _, prev, fut = train_ds[0]
        inp_flat = LoadBatch(prev)
        if isinstance(inp_flat, np.ndarray):
            inp_flat = torch.from_numpy(inp_flat).float().to(device)
        else:
            inp_flat = inp_flat.to(device)
        model.eval()
        with torch.no_grad():
            try:
                pred_flat = model.generate(inp_flat, pred_len=args.pred_len)
            except Exception:
                pred_flat = model(inp_flat, tgt=None, teacher_forcing_ratio=0.0)
        pred_np = pred_flat.cpu().numpy()
        # save sample and true to .mat
        scio.savemat(os.path.join(args.save_dir, 'sample_pred.mat'), {'pred': pred_np, 'true': LoadBatch(fut)})
    except Exception as e:
        print("Could not create sample pred:", e)

    return


if __name__ == '__main__':
    main()
