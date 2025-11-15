#!/usr/bin/env python3
"""
test_hybrid.py

Test / evaluate the HybridTransformerLSTM on your CDL-B test dataset.
Produces prediction plot, NMSE and achievable-rate curves, and saves .mat results.

Assumptions:
- SeqData, LoadBatch, NMSELoss, real2complex, get_zf_rate exist in repo (same as in your other test scripts).
- models.hybrid.HybridTransformerLSTM exists (we added it).
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as scio

# repo utilities (same names as your project)
from data import SeqData, LoadBatch
from metrics import NMSELoss
from utils import real2complex, get_zf_rate  # adapt if utils names differ
from models.hybrid import HybridTransformerLSTM

# try to silence MKL duplicate lib error (your repo does that)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type=bool, default=0)
    parser.add_argument('--gpu_list', type=str, default='0')
    parser.add_argument('--hybrid_ckpt', type=str, default=None, help='path to hybrid checkpoint (optional)')

    parser.add_argument('--seq_len', type=int, default=25)
    parser.add_argument('--pred_len', type=int, default=5)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--ir_test', type=int, default=1)
    parser.add_argument('--v_max', type=int, default=60)
    parser.add_argument('--v_min', type=int, default=30)

    # model dims (keep defaults consistent with earlier scripts)
    parser.add_argument('--enc_in', type=int, default=16, help='input size (flattened real/imag per time-step)')
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--hs', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--hl', type=int, default=2, help='LSTM num layers')

    parser.add_argument('--SNR', type=float, default=14.0)

    return parser.parse_args()


def ensure_results_dir(path='./results'):
    os.makedirs(path, exist_ok=True)
    return path


def main():
    args = parse_args()

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
    device = torch.device('cuda:0') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    print("Using device:", device)

    # load test data (same call signature you used earlier)
    testpath = 'CDL-B/test'
    testData = SeqData(
        testpath,
        prev_len=args.seq_len,
        pred_len=args.pred_len,
        mode='test',
        SNR=args.SNR,
        ir=args.ir_test,
        samples=args.samples,
        v_min=args.v_min,
        v_max=args.v_max
    )

    # quick check
    data_example, _, data_prev, data_pred = testData[0]
    L, M, Nr, Nt = data_example.shape
    print(f"Sample shape (L,M,Nr,Nt) = ({L},{M},{Nr},{Nt})")

    # instantiate hybrid model
    hybrid = HybridTransformerLSTM(
        enc_in=args.enc_in,
        d_model=args.d_model,
        nhead=max(1, args.n_heads // 2),
        trans_layers=max(1, args.e_layers // 2),
        lstm_hidden=args.hs,
        lstm_layers=max(1, args.hl),
        pred_len=args.pred_len,
        dropout=0.05
    )

    # load checkpoint if given
    if args.hybrid_ckpt is not None:
        try:
            print("Loading hybrid checkpoint:", args.hybrid_ckpt)
            hybrid.load_checkpoint(args.hybrid_ckpt, map_location='cpu', strict=True)
            print("Loaded hybrid checkpoint successfully.")
        except Exception as e:
            print("Warning: could not load hybrid checkpoint:", e)
            # try non-strict load
            try:
                hybrid.load_checkpoint(args.hybrid_ckpt, map_location='cpu', strict=False)
                print("Loaded hybrid checkpoint with strict=False.")
            except Exception as e2:
                print("Failed to load checkpoint even with strict=False; continuing with random init. Error:", e2)

    # move to device and dataparallel if requested
    hybrid = torch.nn.DataParallel(hybrid).cuda() if (device.type == 'cuda') else hybrid.to(device)

    hybrid.eval()

    # prepare accumulators
    pred_slots = args.pred_len * args.ir_test + 1
    Rate_hybrid = np.zeros(pred_slots)
    NMSE_hybrid = np.zeros(pred_slots)

    criterion = NMSELoss()
    SNR = args.SNR
    N_it = len(testData)
    print("Number of test samples:", N_it)

    with torch.no_grad():
        for it in range(N_it):
            data, _, inp, label_net = testData[it]

            # data: [L, M, Nr, Nt]
            T, M_local, Nr_local, Nt_local = data.shape
            assert M_local == M and Nr_local == Nr and Nt_local == Nt, "Data dims inconsistent."

            # rearrange data like your other script: transpose -> [M, T, Nr, Nt]
            data_np = np.array(data)
            data_np = data_np.transpose([1, 0, 2, 3])  # [M, T, Nr, Nt]
            label = data_np[:, -args.pred_len:, ...]   # [M, pred_len, Nr, Nt]

            # prepare network inputs (LoadBatch will convert to flattened real/imag representation like your other script)
            inp_net, label_net = LoadBatch(inp), LoadBatch(label_net)
            # ensure tensors
            if isinstance(inp_net, np.ndarray):
                inp_net = torch.from_numpy(inp_net).float()
            if isinstance(label_net, np.ndarray):
                label_net = torch.from_numpy(label_net).float()

            inp_net = inp_net.to(device)      # [B= M, seq_len, enc_in]
            label_net = label_net.to(device)  # [B= M, pred_len, enc_in]

            # run inference (use generate to avoid teacher forcing)
            # hybrid might be wrapped in DataParallel; call accordingly
            if isinstance(hybrid, torch.nn.DataParallel):
                outputs_hybrid = hybrid.module.generate(inp_net, pred_len=args.pred_len)
            else:
                outputs_hybrid = hybrid.generate(inp_net, pred_len=args.pred_len)

            outputs_hybrid = outputs_hybrid.cpu().detach()  # [M, pred_len, enc_in]
            nmse_val = criterion(outputs_hybrid, label_net.cpu())  # criterion expects same tensor formats
            # convert to complex shape for downstream metrics and plotting
            outputs_hybrid_c = real2complex(np.array(outputs_hybrid))  # [M, pred_len, Nr, Nt] if LoadBatch encoded that way

            # if outputs_hybrid_c didn't already have shape [M, pred_len, Nr, Nt] we must reshape.
            # Many of your scripts expect reshape as: outputs.reshape([M, pred_len, Nr, Nt])
            try:
                outputs_hybrid_c = outputs_hybrid_c.reshape([M, args.pred_len, Nr, Nt])
            except Exception:
                # nothing to do if already shaped
                pass

            # PVEC and PAD not needed here; compute NMSE and rate arrays for hybrid
            # For s=0 (ideal) we compare last observed frame to itself; keep consistent with your original scripts
            for s in range(args.pred_len + 1):
                H_true = label[:, s - 1, :, :] if s > 0 else data_np[:, -args.pred_len - 1, ...]
                H_hat = outputs_hybrid_c[:, s - 1, :, :] if s > 0 else data_np[:, -args.pred_len - 1, ...]
                Rate_hybrid[s] += get_zf_rate(H_hat, H_true, SNR) / N_it
                error = np.sum(np.abs(H_true - H_hat) ** 2)
                power = np.sum(np.abs(H_true) ** 2)
                NMSE_hybrid[s] += error / power / N_it

            print('[{}/{}] nmse_hybrid: {:.6f}'.format(it + 1, N_it, nmse_val))

    # plotting & save
    results_dir = ensure_results_dir('./results')

    # pred example: pick first subcarrier and first Rx/TX antenna
    plt.figure()
    x = np.arange(data_np.shape[1])  # time slots axis (T)
    plt.plot(x, data_np[0, :, 0, 0], '--', label='true trace')  # subcarrier=0, antenna indices
    plt.plot(x[-args.pred_len:], outputs_hybrid_c[0, :, 0, 0], label='hybrid pred')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'pred_hybrid.png'))
    plt.close()

    # rate plot
    plt.figure()
    eps = 1e-12
    plt.plot(Rate_hybrid, label='Hybrid')
    plt.xlabel('SRS (0.625 ms)')
    plt.ylabel('Average rate (bit/s/Hz)')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'rate_hybrid.png'))
    plt.close()

    # NMSE plot (dB)
    plt.figure()
    plt.plot(10 * np.log10(NMSE_hybrid + eps), label='Hybrid')
    plt.xlabel('SRS (0.625 ms)')
    plt.ylabel('NMSE (dB)')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'NMSE_hybrid.png'))
    plt.close()

    # save .mat outputs for later comparison
    scio.savemat(os.path.join(results_dir, 'CR_hybrid_{}.mat'.format(args.v_max)), {'data': outputs_hybrid_c})
    scio.savemat(os.path.join(results_dir, 'NMSE_Hybrid_{}.mat'.format(args.v_max)), {'NMSE': NMSE_hybrid})
    scio.savemat(os.path.join(results_dir, 'Rate_Hybrid_{}.mat'.format(args.v_max)), {'rate': Rate_hybrid})

    print("Saved results into", results_dir)
    print("Done.")


if __name__ == '__main__':
    main()
