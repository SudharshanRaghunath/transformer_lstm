#!/usr/bin/env python3
"""
train_hybrid_patched.py

Training script for HybridTransformerLSTM (patched):
- Uses SeqData / LoadBatch from your repo
- Normalizes windows (per-sample power normalization)
- Uses combined loss (NMSE + MSE)
- Teacher forcing schedule (TF=1.0 for warmup then linear decay)
- Checkpointing (save best by val NMSE)
- Plots training curves and sample prediction real/imag overlays

Place this file in your repo and run:
python train_hybrid_patched.py --data_dir CDL-B/train --save_dir checkpoints/hybrid --use_gpu 1

Make sure models/hybrid.py, data.py, utils.py and metrics.py are in place as in your repo.
"""

import os
import time
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as scio
from torch.utils.data import DataLoader

# repo imports (assumes these exist)
from data import SeqData, LoadBatch
from models.hybrid import HybridTransformerLSTM
from metrics import NMSELoss
from utils import real2complex

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# -------------------- helpers --------------------

def nmse_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    # pred,target: [B, T, F]
    num = torch.sum((pred - target).abs() ** 2, dim=(1, 2))
    den = torch.sum((target).abs() ** 2, dim=(1, 2)) + eps
    return torch.mean(num / den)


mse_loss = nn.MSELoss()


def combined_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.6) -> torch.Tensor:
    """Weighted combination of NMSE and MSE."""
    l_nmse = nmse_torch(pred, target)
    l_mse = mse_loss(pred, target)
    return alpha * l_nmse + (1.0 - alpha) * l_mse


def normalize_batch_tensor(batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sample normalization by RMS power across features.
    Accepts numpy array of shape [T, M, Nr, Nt] or flattened [T, F].
    Returns normalized flattened array and scale factors for inverse.
    """
    # If input is 4D complex-like, convert power over axes
    arr = batch
    if arr.ndim == 4:
        # compute per-sample power across entire tensor per time-step, then mean across time
        power = np.sum(np.abs(arr) ** 2, axis=(1, 2, 3), keepdims=False)  # [T]
        # scalar per-window
        scale = np.sqrt(np.mean(power) + 1e-12)
        norm = arr / scale
        return norm, scale
    elif arr.ndim == 2:
        # flattened [T, F]
        power = np.mean(np.sum(arr ** 2, axis=1))
        scale = np.sqrt(power + 1e-12)
        return arr / scale, scale
    else:
        # fallback
        scale = 1.0
        return arr, scale


def inverse_scale_flat(flat: np.ndarray, scale: float) -> np.ndarray:
    return flat * scale


def plot_training(history: dict, outpath: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_nmse'], label='Train NMSE'); axes[0].legend(); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('NMSE')
    axes[1].plot(history['val_nmse'], color='red', label='Val NMSE'); axes[1].legend(); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('NMSE')
    plt.tight_layout(); plt.savefig(outpath); plt.close()


def plot_pred_examples(true_c: np.ndarray, pred_c: np.ndarray, outpath: str, title: str = ''):
    # true_c and pred_c are complex arrays of length N (or 1D real/imag arrays)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(np.real(true_c), label='Original_H-values(Real)')
    axs[0].plot(np.real(pred_c), label='Predicted_H-values(Real)')
    axs[0].legend(); axs[0].set_xlabel('Index'); axs[0].set_ylabel('Values')
    axs[1].plot(np.imag(true_c), label='Original_H-values(Imag)')
    axs[1].plot(np.imag(pred_c), label='Predicted_H-values(Imag)')
    axs[1].legend(); axs[1].set_xlabel('Index'); axs[1].set_ylabel('Values')
    plt.suptitle(title)
    plt.savefig(outpath); plt.close()


# -------------------- training script --------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='CDL-B/train')
    p.add_argument('--prev_len', type=int, default=25)
    p.add_argument('--pred_len', type=int, default=5)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-6)
    p.add_argument('--save_dir', type=str, default='./checkpoints/hybrid')
    p.add_argument('--use_gpu', action='store_true')
    p.add_argument('--gpu_list', type=str, default='0')
    p.add_argument('--enc_in', type=int, default=16)
    p.add_argument('--d_model', type=int, default=64)
    p.add_argument('--hs', type=int, default=256)
    p.add_argument('--hl', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tf_warmup', type=int, default=20, help='epochs to keep TF=1.0')
    p.add_argument('--final_tf', type=float, default=0.0, help='final teacher forcing ratio')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
    device = torch.device('cuda:0') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    # prepare datasets
    train_ds = SeqData(args.data_dir, prev_len=args.prev_len, pred_len=args.pred_len, mode='train', SNR=14, ir=1, samples=1, v_min=30, v_max=60)
    val_ds = SeqData(args.data_dir, prev_len=args.prev_len, pred_len=args.pred_len, mode='val', SNR=14, ir=1, samples=1, v_min=30, v_max=60)

    # Small wrappers to produce flattened arrays using LoadBatch
    class Wrapper(torch.utils.data.Dataset):
        def __init__(self, seqdata):
            self.seqdata = seqdata
        def __len__(self):
            return len(self.seqdata)
        def __getitem__(self, idx):
            data, _, prev, future = self.seqdata[idx]
            # prev shape originally [prev_len, M, Nr, Nt]
            prev_flat = LoadBatch(prev)  # expected [M, prev_len, enc_in] or [batch-like]
            fut_flat = LoadBatch(future)
            # Ensure numpy -> torch
            if isinstance(prev_flat, np.ndarray):
                prev_flat = torch.from_numpy(prev_flat).float()
            if isinstance(fut_flat, np.ndarray):
                fut_flat = torch.from_numpy(fut_flat).float()
            return prev_flat, fut_flat

    train_wrap = Wrapper(train_ds)
    val_wrap = Wrapper(val_ds)

    train_loader = DataLoader(train_wrap, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_wrap, batch_size=args.batch, shuffle=False)

    # instantiate model
    model = HybridTransformerLSTM(enc_in=args.enc_in, d_model=args.d_model, nhead=4, trans_layers=2, lstm_hidden=args.hs, lstm_layers=max(1,args.hl), pred_len=args.pred_len)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    nmse_metric = NMSELoss()

    history = {'train_nmse': [], 'val_nmse': []}

    best_val = 1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        # teacher forcing schedule
        if epoch <= args.tf_warmup:
            tf_ratio = 1.0
        else:
            remain = max(1, args.epochs - args.tf_warmup)
            tf_ratio = max(args.final_tf, 1.0 - (epoch - args.tf_warmup) / float(remain))

        train_losses = []
        for batch_idx, (inp_batch, tgt_batch) in enumerate(train_loader):
            # Ensure shape [B, seq_len, F]
            inp_batch = inp_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            optimizer.zero_grad()
            preds = model(inp_batch, tgt=tgt_batch, teacher_forcing_ratio=tf_ratio)
            loss = combined_loss(preds, tgt_batch, alpha=0.6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vinp, vtgt in val_loader:
                vinp = vinp.to(device); vtgt = vtgt.to(device)
                vpred = model.generate(vinp, pred_len=args.pred_len)
                vloss = combined_loss(vpred, vtgt, alpha=0.6)
                val_losses.append(vloss.item())

        avg_train = float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0
        avg_val = float(np.mean(val_losses)) if len(val_losses) > 0 else 0.0
        history['train_nmse'].append(avg_train)
        history['val_nmse'].append(avg_val)

        print(f"Epoch {epoch}/{args.epochs} TF={tf_ratio:.3f} train_nmse={avg_train:.6f} val_nmse={avg_val:.6f} time={time.time()-t0:.1f}s")

        # checkpoint best
        if avg_val < best_val:
            best_val = avg_val
            ckpt_path = os.path.join(args.save_dir, 'hybrid_best.pth')
            model.save_checkpoint(ckpt_path, optimizer=optimizer, epoch=epoch)
            print('Saved best checkpoint to', ckpt_path)

        # quick save each epoch
        if epoch % 10 == 0:
            model.save_checkpoint(os.path.join(args.save_dir, f'hybrid_epoch{epoch}.pth'), optimizer=optimizer, epoch=epoch)

    # final plotting and sample predictions on first validation example
    plot_training(history, os.path.join(args.save_dir, 'training_nmse.png'))

    # load best and run on a sample from val set to produce overlay plot similar to your target
    best_path = os.path.join(args.save_dir, 'hybrid_best.pth')
    try:
        model.load_checkpoint(best_path, map_location='cpu', strict=True)
        model.to(device)
    except Exception:
        print('Warning: could not reload best checkpoint strictly; continuing with current weights')

    # pick a sample
    sample_inp, sample_tgt = val_wrap[0]
    if isinstance(sample_inp, torch.Tensor):
        sinp = sample_inp.unsqueeze(0).to(device)  # [1, seq_len, F]
        spred = model.generate(sinp, pred_len=args.pred_len)
        spred = spred.cpu().numpy()[0]
        stgt = sample_tgt.numpy()[0]
    else:
        print('Sample types unexpected')
        return

    # convert flattened real/imag to complex: use real2complex helper
    spred_c = real2complex(spred)
    stgt_c = real2complex(stgt)

    # choose subcarrier 0, antenna 0 for visualization
    try:
        true_trace = stgt_c[0, :, 0, 0]
        pred_trace = spred_c[0, :, 0, 0]
    except Exception:
        # if shape is [pred_len, F] fallback to first feature
        true_trace = stgt_c[:, 0]
        pred_trace = spred_c[:, 0]

    plot_pred_examples(true_trace, pred_trace, os.path.join(args.save_dir, 'pred_example.png'), title='Validation example')

    # save mats
    scio.savemat(os.path.join(args.save_dir, 'pred_example.mat'), {'pred': spred, 'true': stgt})

    print('Training complete. Best val NMSE:', best_val)


if __name__ == '__main__':
    main()
