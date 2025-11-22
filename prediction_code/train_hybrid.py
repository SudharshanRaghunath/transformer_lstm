#!/usr/bin/env python3
"""
Corrected test_hybrid.py — robust evaluation + diagnostics.
Save as prediction_code/test_hybrid.py (replace your current file).
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as scio
import json
from collections import OrderedDict

from data import SeqData, LoadBatch
from metrics import NMSELoss
from utils import real2complex, get_zf_rate
from models.hybrid import HybridTransformerLSTM

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--use_gpu', type=int, default=0)
    p.add_argument('--gpu_list', type=str, default='0')
    p.add_argument('--hybrid_ckpt', type=str, default=None)
    p.add_argument('--seq_len', type=int, default=25)
    p.add_argument('--pred_len', type=int, default=5)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--ir_test', type=int, default=1)
    p.add_argument('--v_max', type=int, default=60)
    p.add_argument('--v_min', type=int, default=30)
    p.add_argument('--enc_in', type=int, default=16)
    p.add_argument('--d_model', type=int, default=64)
    p.add_argument('--hs', type=int, default=256)
    p.add_argument('--hl', type=int, default=2)
    p.add_argument('--SNR', type=float, default=14.0)
    return p.parse_args()

def ensure_dir(d): os.makedirs(d, exist_ok=True); return d

def dump_debug(txt, results_dir='./results'):
    ensure_dir(results_dir)
    with open(os.path.join(results_dir,'test_hybrid_debug.txt'),'a') as f:
        f.write(txt + '\n')

def load_checkpoint_verbose(model, path, device):
    """Load checkpoint and print missing/unexpected keys (helpful for debugging)."""
    raw = torch.load(path, map_location=device)
    state = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw
    try:
        # attempt strict load first
        model.load_state_dict(state, strict=True)
        print("Checkpoint loaded (strict=True). All keys matched.")
        dump_debug("Checkpoint loaded (strict=True): " + str(path))
        return True
    except Exception as e:
        print("Strict load failed:", e)
        dump_debug("Strict load failed: " + str(e))
        # attempt non-strict and report missing/unexpected
        model_dict = model.state_dict()
        loaded_keys = set(state.keys())
        model_keys = set(model_dict.keys())
        missing = model_keys - loaded_keys
        unexpected = loaded_keys - model_keys
        print(f"Missing keys in checkpoint: {len(missing)}")
        print(f"Unexpected keys in checkpoint: {len(unexpected)}")
        dump_debug(f"Missing keys ({len(missing)}): {list(sorted(missing))}")
        dump_debug(f"Unexpected keys ({len(unexpected)}): {list(sorted(unexpected))}")
        # try non-strict load
        model.load_state_dict(state, strict=False)
        print("Checkpoint loaded with strict=False (missing keys were left random).")
        dump_debug("Checkpoint loaded with strict=False")
        return False

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
    device = torch.device('cuda:0') if (args.use_gpu and torch.cuda.is_available()) else torch.device('cpu')
    print("Using device:", device); dump_debug(f"Using device: {device}")

    testpath = 'CDL-B/test'
    if not os.path.isdir(testpath):
        raise FileNotFoundError(f"Test data path not found: {testpath}. Please set --data_dir correctly or create CDL-B/test")

    testData = SeqData(testpath, prev_len=args.seq_len, pred_len=args.pred_len,
                       mode='test', SNR=args.SNR, ir=args.ir_test, samples=1,
                       v_min=args.v_min, v_max=args.v_max)

    # instantiate hybrid
    hybrid = HybridTransformerLSTM(
        enc_in=args.enc_in,
        d_model=args.d_model,
        nhead=4,
        trans_layers=2,
        lstm_hidden=args.hs,
        lstm_layers=max(1,args.hl),
        pred_len=args.pred_len,
        dropout=0.05
    )

    # load checkpoint (if provided)
    loaded_ok = False
    if args.hybrid_ckpt:
        if not os.path.isfile(args.hybrid_ckpt):
            print("Checkpoint path not found:", args.hybrid_ckpt)
            dump_debug("Checkpoint not found: " + str(args.hybrid_ckpt))
        else:
            loaded_ok = load_checkpoint_verbose(hybrid, args.hybrid_ckpt, device)

    # move to device BEFORE wrapping in DataParallel
    hybrid.to(device)
    if device.type == 'cuda' and torch.cuda.is_available():
        hybrid = torch.nn.DataParallel(hybrid).cuda()
    hybrid.eval()

    # accumulators
    pred_slots = args.pred_len * args.ir_test + 1
    Rate_hybrid = np.zeros(pred_slots)
    NMSE_hybrid = np.zeros(pred_slots)
    prev_nmse = np.zeros(pred_slots)  # baseline (previous)
    prev_rate = np.zeros(pred_slots)

    criterion = NMSELoss()
    SNR = args.SNR
    N_it = len(testData)
    print("Number test samples:", N_it); dump_debug(f"N_test: {N_it}")

    # quick baseline: compute NMSE of previous (last observed) for comparison
    with torch.no_grad():
        for it in range(N_it):
            data, _, inp, label_net = testData[it]   # your SeqData returns (data,_,inp,label_net)
            # convert shapes
            data_np = np.array(data).transpose([1,0,2,3])  # [M, T, Nr, Nt]
            label = data_np[:, -args.pred_len:, ...]       # [M, pred_len, Nr, Nt]
            prev = data_np[:, -args.pred_len-1, ...]      # previous frame repeated
            for s in range(args.pred_len+1):
                H_true = label[:, s-1, :, :] if s>0 else prev
                H_hat_prev = prev
                prev_rate[s] += get_zf_rate(H_hat_prev, H_true, SNR) / N_it
                err = np.sum(np.abs(H_true - H_hat_prev)**2)
                pwr = np.sum(np.abs(H_true)**2)
                prev_nmse[s] += err/pwr / N_it

    # now hybrid inference
    with torch.no_grad():
        for it in range(N_it):
            data, _, inp, label_net = testData[it]
            # check shapes
            T, M, Nr, Nt = np.array(data).shape
            data_np = np.array(data).transpose([1,0,2,3])  # [M, T, Nr, Nt]
            label = data_np[:, -args.pred_len:, ...]

            # convert inp and label_net via LoadBatch (same as training pipeline)
            inp_net = LoadBatch(inp)
            label_net_t = LoadBatch(label_net)
            if isinstance(inp_net, np.ndarray):
                inp_t = torch.from_numpy(inp_net).float().to(device)
            else:
                inp_t = inp_net.to(device)
            if isinstance(label_net_t, np.ndarray):
                label_t = torch.from_numpy(label_net_t).float().to(device)
            else:
                label_t = label_net_t.to(device)

            # shapes check
            print(f"it={it} inp shape {inp_t.shape} label shape {label_t.shape}")
            dump_debug(f"it={it} inp shape {inp_t.shape} label shape {label_t.shape}")

            # generate predictions
            if isinstance(hybrid, torch.nn.DataParallel):
                preds = hybrid.module.generate(inp_t, pred_len=args.pred_len)
            else:
                preds = hybrid.generate(inp_t, pred_len=args.pred_len)

            preds = preds.cpu().numpy()  # [M, pred_len, enc_in] or [B, pred_len, enc_in]
            # compute nmse on flattened representation (same format as criterion expects)
            # convert preds and label back to same space for criterion: use torch tensors
            preds_t = torch.from_numpy(preds).float()
            label_cpu = label_t.cpu().float()
            # if dims mismatch, try to reshape: expected [M, pred_len, enc_in]
            try:
                nmse_val = criterion(preds_t, label_cpu).item()
            except Exception as e:
                print("NMSE criterion failed:", e)
                dump_debug("NMSE criterion failed: " + str(e))
                nmse_val = None

            # convert preds into complex shape used by earlier scripts
            try:
                preds_c = real2complex(preds)
                preds_c = preds_c.reshape([M, args.pred_len, Nr, Nt])
            except Exception as e:
                print("real2complex/reshape failed:", e)
                dump_debug("real2complex failed: " + str(e))
                # fallback: attempt shape inference
                preds_c = preds

            # compute NMSE and rate per time step s (s=0..pred_len)
            for s in range(args.pred_len+1):
                H_true = label[:, s-1, :, :] if s>0 else data_np[:, -args.pred_len-1, ...]
                H_hat = preds_c[:, s-1, :, :] if s>0 else data_np[:, -args.pred_len-1, ...]
                Rate_hybrid[s] += get_zf_rate(H_hat, H_true, SNR) / N_it
                err = np.sum(np.abs(H_true - H_hat)**2)
                pwr = np.sum(np.abs(H_true)**2)
                NMSE_hybrid[s] += err/pwr / N_it

            print(f"[{it+1}/{N_it}] nmse(flat)={nmse_val} nmse_hybrid_step0={NMSE_hybrid[1] if args.pred_len>=1 else NMSE_hybrid[0]}")
            dump_debug(f"[{it+1}/{N_it}] nmse(flat)={nmse_val}")

    # plots
    results_dir = ensure_dir('./results')
    x = np.arange(args.pred_len+1)
    plt.figure(); plt.plot(10*np.log10(prev_nmse + 1e-12), '--', label='Previous (baseline)')
    plt.plot(10*np.log10(NMSE_hybrid + 1e-12), label='Hybrid')
    plt.xlabel('SRS (0.625 ms)'); plt.ylabel('NMSE (dB)'); plt.legend(); plt.savefig(os.path.join(results_dir,'NMSE_hybrid_vs_prev.png')); plt.close()

    plt.figure(); plt.plot(prev_rate, '--', label='Previous'); plt.plot(Rate_hybrid, label='Hybrid'); plt.xlabel('SRS'); plt.ylabel('Rate'); plt.legend(); plt.savefig(os.path.join(results_dir,'Rate_hybrid_vs_prev.png')); plt.close()

    # pred example for first subcarrier/antenna
    try:
        plt.figure()
        t_axis = np.arange(data_np.shape[1])
        plt.plot(t_axis, data_np[0,:,0,0], '--', label='true trace')
        plt.plot(t_axis[-args.pred_len:], preds_c[0,:,0,0], label='hybrid pred')
        plt.legend(); plt.savefig(os.path.join(results_dir,'pred_hybrid_example.png')); plt.close()
    except Exception as e:
        print("example plot failed:", e); dump_debug("example plot failed: " + str(e))

    # per-subcarrier NMSE heatmap
    # (this is optional debug, compute NMSE per subcarrier over prediction horizon)
    try:
        per_sub_nmse = np.zeros((M, args.pred_len))
        for m in range(M):
            for s in range(args.pred_len):
                Ht = label[m,s,...]
                Hp = preds_c[m,s,...]
                per_sub_nmse[m,s] = np.sum(np.abs(Ht-Hp)**2) / (np.sum(np.abs(Ht)**2) + 1e-12)
        plt.figure(figsize=(6,10)); plt.imshow(10*np.log10(per_sub_nmse+1e-12), aspect='auto', origin='lower'); plt.colorbar(); plt.title('per-subcarrier NMSE(dB)'); plt.savefig(os.path.join(results_dir,'per_subcarrier_nmse.png')); plt.close()
    except Exception as e:
        print("per-subcarrier NMSE failed:", e); dump_debug("per-subcarrier NMSE failed: " + str(e))

    # save mats
    scio.savemat(os.path.join(results_dir, f'NMSE_hybrid_{args.v_max}.mat'), {'NMSE': NMSE_hybrid, 'NMSE_prev': prev_nmse})
    scio.savemat(os.path.join(results_dir, f'Rate_hybrid_{args.v_max}.mat'), {'Rate': Rate_hybrid, 'Rate_prev': prev_rate})

    print("Done. Results and debug log in ./results/test_hybrid_debug.txt")

if __name__=='__main__':
    main()
