import os
from utils import *
import argparse
from torch.utils.data import DataLoader
from metrics import NMSELoss
from models.model import InformerStack, LSTM, RNN, GRU, InformerStack_e2e
import matplotlib.pyplot as plt
from pvec import pronyvec
from PAD import PAD3
import scipy.io as scio
import torch
import torch.nn as nn
import numpy as np

from data import SeqData, LoadBatch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='0')
parser.add_argument('--use_gpu', type=bool, default=0)
parser.add_argument('--gpu_list', type=str, default='1', help='input gpu list')

parser.add_argument('--SNR', type=float, default=10)

parser.add_argument('--seq_len', type=int, default=25, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=10, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=5, help='prediction sequence length')

parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--samples', type=int, default=1)
parser.add_argument('--ir_test', type=int, default=1)
parser.add_argument('--v_max', type=int, default=60)
parser.add_argument('--v_min', type=int, default=30)

# LSTM
parser.add_argument('--hs', type=int, default=256)
parser.add_argument('--hl', type=int, default=2)
# informer
parser.add_argument('--enc_in', type=int, default=16, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=16, help='decoder input size')
parser.add_argument('--c_out', type=int, default=16, help='output size')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=3, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='full', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='fixed',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# Hybrid checkpoint (optional)
parser.add_argument('--hybrid_ckpt', type=str, default=None, help='path to hybrid model checkpoint (optional)')

args = parser.parse_args()

informer_settings_e2e = '{}_data_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}'.format(
    'informerstack_e2e', args.data,
    args.seq_len, args.label_len, args.pred_len,
    64, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil)

informer_settings = '{}_data_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}'.format('informerstack',
                                                                                                    args.data,
                                                                                                    args.seq_len,
                                                                                                    args.label_len,
                                                                                                    args.pred_len,
                                                                                                    64, args.n_heads,
                                                                                                    args.e_layers,
                                                                                                    args.d_layers,
                                                                                                    args.d_ff,
                                                                                                    args.attn,
                                                                                                    args.factor,
                                                                                                    args.embed,
                                                                                                    args.distil)

lstm_settings = '{}_data_{}_sl{}_pl{}_hs{}_hl{}'.format('LSTM', args.data,
                                                        args.seq_len, args.pred_len, args.hs, args.hl)
gru_settings = '{}_data_{}_sl{}_pl{}_hs{}_hl{}'.format('GRU', args.data,
                                                       args.seq_len, args.pred_len, args.hs, args.hl)
rnn_settings = '{}_data_{}_sl{}_pl{}_hs{}_hl{}'.format('RNN', args.data,
                                                       args.seq_len, args.pred_len, args.hs, args.hl)

print(informer_settings)
print(informer_settings_e2e)
print(lstm_settings)
print(gru_settings)
print(rnn_settings)
# Parameters Setting for Training


gpu_list = args.gpu_list

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device('cuda:0') if args.use_gpu else torch.device('cpu')

# ---------- Hybrid model definition ----------
class HybridTransformerLSTM(nn.Module):
    """
    Hybrid model that accepts enc_inp shaped [batch, seq_len, enc_in]
    and predicts [batch, pred_len, enc_in].
    Architecture:
      - Input linear -> d_model
      - Positional embedding
      - TransformerEncoder over time dimension (seq_len)
      - LSTM decoder autoregressively predicting pred_len frames
      - Linear head to map back to enc_in
    """
    def __init__(self, enc_in, d_model=64, nhead=4, num_layers=2, lstm_hidden=256, lstm_layers=1, pred_len=5, dropout=0.05):
        super().__init__()
        self.enc_in = enc_in
        self.d_model = d_model
        self.input_proj = nn.Linear(enc_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # simple learnable time positional embedding
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, d_model))  # supports up to 1024 timesteps
        # LSTM decoder
        self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.dec_proj = nn.Linear(lstm_hidden, enc_in)
        self.pred_len = pred_len
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

    def forward(self, enc_inp):
        # enc_inp: [b, seq_len, enc_in]
        b, seq_len, enc_in = enc_inp.shape
        x = self.input_proj(enc_inp)  # [b, seq_len, d_model]
        # add pos emb (slice as needed)
        if seq_len <= self.pos_emb.size(1):
            x = x + self.pos_emb[:, :seq_len, :]
        else:
            # if seq longer than preset, fallback to zeros (rare for current use)
            x = x
        x_enc = self.transformer(x)  # [b, seq_len, d_model]
        # seed decoder with last encoder output
        dec_input = x_enc[:, -1:, :]  # [b,1,d_model]
        # initialize hidden
        h0 = torch.zeros(self.lstm_layers, b, self.lstm_hidden, device=x.device)
        c0 = torch.zeros(self.lstm_layers, b, self.lstm_hidden, device=x.device)
        preds = []
        hidden = (h0, c0)
        cur = dec_input
        for _ in range(self.pred_len):
            out, hidden = self.lstm(cur, hidden)  # out: [b,1,lstm_hidden]
            frame = self.dec_proj(out.squeeze(1))  # [b, enc_in]
            preds.append(frame.unsqueeze(1))
            # feedback: project frame back into d_model to feed next LSTM step
            cur = self.input_proj(frame.unsqueeze(1))
        preds = torch.cat(preds, dim=1)  # [b, pred_len, enc_in]
        return preds

# --------------------------------------------

# Data loading
visualize = 1

print('Set your test set directory')
testpath = 'CDL-B/test'

testData = SeqData(
    testpath,
    prev_len=args.seq_len,
    pred_len=args.pred_len,
    mode='test',
    SNR=14,
    ir=args.ir_test,
    samples=args.samples,
    v_min=args.v_min,
    v_max=args.v_max
)

test_loader = DataLoader(dataset=testData, batch_size=args.batch, shuffle=False,
                         num_workers=4., drop_last=False, pin_memory=True)

data, _, data_prev, data_pred = testData[0]
L, M, Nr, Nt = data.shape

informer = InformerStack(
    args.enc_in,
    args.dec_in,
    args.c_out,
    args.seq_len,
    args.label_len,
    args.pred_len,
    args.factor,
    64,  # args.d_model,
    args.n_heads,
    args.e_layers,
    args.d_layers,
    args.d_ff,
    args.dropout,
    args.attn,
    args.embed,
    args.activation,
    args.output_attention,
    args.distil,
    device
)

# model structure
informer_e2e = InformerStack_e2e(
    args.enc_in,
    args.dec_in,
    args.c_out,
    args.seq_len,
    args.label_len,
    args.pred_len,
    args.factor,
    64,
    args.n_heads,
    args.e_layers,
    args.d_layers,
    args.d_ff,
    args.dropout,
    args.attn,
    args.embed,
    args.activation,
    args.output_attention,
    args.distil,
    device
)

lstm = LSTM(args.enc_in, args.enc_in, args.hs, args.hl)
rnn = RNN(args.enc_in, args.enc_in, args.hs, args.hl)
gru = GRU(args.enc_in, args.enc_in, args.hs, args.hl)

# instantiate hybrid
hybrid = HybridTransformerLSTM(enc_in=args.enc_in, d_model=args.d_model, nhead=max(1, args.n_heads//2),
                              num_layers=2, lstm_hidden=args.hs, lstm_layers=max(1, args.hl//2),
                              pred_len=args.pred_len, dropout=args.dropout)

# transformer model checkpoints path
path = 'checkpoints/checkpoints_30-60_L5/'

# load informer
informer_path = path + informer_settings
model_load = informer_path + '/checkpoint.pth'
state_dicts = torch.load(model_load, map_location=torch.device('cpu'))
informer.load_state_dict(state_dicts['state_dict'])
print("informer  has been loaded!")
informer = torch.nn.DataParallel(informer).cuda() if args.use_gpu else informer

# load informer_e2e
informer_path = path + informer_settings_e2e
model_load = informer_path + '/checkpoint.pth'
state_dicts = torch.load(model_load, map_location=torch.device('cpu'))
informer_e2e.load_state_dict(state_dicts['state_dict'])
print("informer_e2e  has been loaded!")
informer_e2e = torch.nn.DataParallel(informer_e2e).cuda() if args.use_gpu else informer_e2e

# load LSTM
lstm_path = path + lstm_settings
model_load = lstm_path + '/checkpoint.pth'
state_dicts = torch.load(model_load, map_location=torch.device('cpu'))
lstm.load_state_dict(state_dicts['state_dict'])
print("lstm  has been loaded!")
lstm = torch.nn.DataParallel(lstm).cuda() if args.use_gpu else lstm

# load gru
gru_path = path + gru_settings
model_load = gru_path + '/checkpoint.pth'
state_dicts = torch.load(model_load, map_location=torch.device('cpu'))
gru.load_state_dict(state_dicts['state_dict'])
print("gru  has been loaded!")
gru = torch.nn.DataParallel(gru).cuda() if args.use_gpu else gru

# load rnn
rnn_path = path + rnn_settings
model_load = rnn_path + '/checkpoint.pth'
state_dicts = torch.load(model_load, map_location=torch.device('cpu'))
rnn.load_state_dict(state_dicts['state_dict'])
print("rnn  has been loaded!")
rnn = torch.nn.DataParallel(rnn).cuda() if args.use_gpu else rnn

# try loading hybrid checkpoint if provided
if args.hybrid_ckpt is not None:
    try:
        print("Loading hybrid checkpoint from", args.hybrid_ckpt)
        sd = torch.load(args.hybrid_ckpt, map_location=torch.device('cpu'))
        hybrid.load_state_dict(sd['state_dict'] if 'state_dict' in sd else sd)
        print("Hybrid checkpoint loaded.")
    except Exception as e:
        print("Could not load hybrid checkpoint:", e)

# move hybrid to device
hybrid = torch.nn.DataParallel(hybrid).cuda() if args.use_gpu else hybrid

# set eval
informer.eval()
informer_e2e.eval()
lstm.eval()
gru.eval()
rnn.eval()
hybrid.eval()

# prepare accumulators, include hybrid
pred_slots = args.pred_len * args.ir_test + 1
Rate0 = np.zeros(pred_slots)
Rate1 = np.zeros(pred_slots)
Rate2 = np.zeros(pred_slots)
Rate3 = np.zeros(pred_slots)
Rate4 = np.zeros(pred_slots)
Rate5 = np.zeros(pred_slots)
Rate6 = np.zeros(pred_slots)
Rate7 = np.zeros(pred_slots)
Rate8 = np.zeros(pred_slots)
Rate_hybrid = np.zeros(pred_slots)

NMSE0 = np.zeros(pred_slots)
NMSE1 = np.zeros(pred_slots)
NMSE2 = np.zeros(pred_slots)
NMSE3 = np.zeros(pred_slots)
NMSE4 = np.zeros(pred_slots)
NMSE5 = np.zeros(pred_slots)
NMSE6 = np.zeros(pred_slots)
NMSE7 = np.zeros(pred_slots)
NMSE8 = np.zeros(pred_slots)
NMSE_hybrid = np.zeros(pred_slots)

criterion = NMSELoss()
SNR = 14
N_it = len(testData)

with torch.no_grad():
    for it in range(N_it):
        data, _, inp, label_net = testData[it]

        T, M, Nr, Nt = data.shape
        data = np.array(data)
        data = (data.transpose([1, 0, 2, 3]))
        label = data[:, -args.pred_len:, ...]

        inp_net, label_net = LoadBatch(inp), LoadBatch(label_net)
        inp_net = inp_net.to(device)

        inp = inp.transpose([1, 0, 2, 3])
        enc_inp = inp_net
        dec_inp = torch.zeros_like(inp_net[:, -args.pred_len:, :]).to(device)
        dec_inp = torch.cat([inp_net[:, args.seq_len - args.label_len:args.seq_len, :], dec_inp], dim=1)

        # informer
        if args.output_attention:
            outputs_informer = informer(enc_inp, dec_inp)[0]
        else:
            outputs_informer = informer(enc_inp, dec_inp)
        outputs_informer = outputs_informer.cpu().detach()
        nmse_informer = criterion(outputs_informer, label_net)
        outputs_informer = real2complex(np.array(outputs_informer))

        # informer_e2e
        if args.output_attention:
            outputs_informer_e2e = informer_e2e(enc_inp, dec_inp)[0]
        else:
            outputs_informer_e2e = informer_e2e(enc_inp, dec_inp)
        outputs_informer_e2e = outputs_informer_e2e.cpu().detach()
        nmse_informer_e2e = criterion(outputs_informer_e2e, label_net)
        outputs_informer_e2e = real2complex(np.array(outputs_informer_e2e))

        # lstm
        outputs_lstm = lstm.test_data(enc_inp, args.pred_len, device)
        outputs_lstm = outputs_lstm.cpu().detach()
        nmse_lstm = criterion(outputs_lstm, label_net)
        outputs_lstm = real2complex(np.array(outputs_lstm))

        # gru
        outputs_gru = gru.test_data(enc_inp, args.pred_len, device)
        outputs_gru = outputs_gru.cpu().detach()
        nmse_gru = criterion(outputs_gru, label_net)
        outputs_gru = real2complex(np.array(outputs_gru))

        # rnn
        outputs_rnn = rnn.test_data(enc_inp, args.pred_len, device)
        outputs_rnn = outputs_rnn.cpu().detach()
        nmse_rnn = criterion(outputs_rnn, label_net)
        outputs_rnn = real2complex(np.array(outputs_rnn))

        # hybrid
        outputs_hybrid = hybrid(enc_inp)  # [B, pred_len, enc_in]
        outputs_hybrid = outputs_hybrid.cpu().detach()
        nmse_hybrid_val = criterion(outputs_hybrid, label_net)
        outputs_hybrid = real2complex(np.array(outputs_hybrid))

        # freq AR (PVEC and PAD)
        inp_AR = inp.reshape([M, args.seq_len, Nr * Nt])
        outputs_AR_freq = pronyvec(inp_AR, p=6, startidx=args.seq_len, subcarriernum=M, Nr=Nr, pre_len=args.pred_len)
        outputs_AR_delay = PAD3(inp_AR, p=6, startidx=args.seq_len, subcarriernum=M, Nr=Nr, pre_len=args.pred_len)

        # reshape
        outputs_informer = outputs_informer.reshape([M, args.pred_len, Nr, Nt])
        outputs_informer_e2e = outputs_informer_e2e.reshape([M, args.pred_len, Nr, Nt])
        outputs_lstm = outputs_lstm.reshape([M, args.pred_len, Nr, Nt])
        outputs_gru = outputs_gru.reshape([M, args.pred_len, Nr, Nt])
        outputs_rnn = outputs_rnn.reshape([M, args.pred_len, Nr, Nt])
        outputs_AR_freq = outputs_AR_freq.reshape([M, args.pred_len, Nr, Nt])
        outputs_AR_delay = outputs_AR_delay.reshape([M, args.pred_len, Nr, Nt])
        outputs_hybrid = outputs_hybrid.reshape([M, args.pred_len, Nr, Nt])

        # compute metrics for all methods including hybrid
        for s in range(args.pred_len + 1):
            H_true = label[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            H_hat = label[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            Rate0[s] += get_zf_rate(H_hat, H_true, SNR) / N_it
            error = np.sum(np.abs(H_true - H_hat) ** 2)
            power = np.sum(np.abs(H_true) ** 2)
            NMSE0[s] += error / power / N_it

        # informer
        for s in range(args.pred_len + 1):
            H_true = label[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            H_hat = outputs_informer[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            Rate8[s] += get_zf_rate(H_hat, H_true, SNR) / N_it
            error = np.sum(np.abs(H_true - H_hat) ** 2)
            power = np.sum(np.abs(H_true) ** 2)
            NMSE8[s] += error / power / N_it

        # informer_e2e
        for s in range(args.pred_len + 1):
            H_true = label[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            out_hat = outputs_informer_e2e[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            out_hat = out_hat / np.linalg.norm(out_hat, axis=(1), keepdims=True)
            Heff = np.matmul(out_hat.transpose(0, 2, 1), H_true)
            rate = np.mean(np.log2(
                np.linalg.det(np.eye(2) + (10 ** (SNR / 10)) * np.matmul(Heff.conj().transpose(0, 2, 1), Heff))))
            Rate1[s] += rate / N_it
            error = np.sum(np.abs(H_true - out_hat) ** 2)
            power = np.sum(np.abs(H_true) ** 2)
            NMSE1[s] += error / power / N_it

        # lstm
        for s in range(args.pred_len + 1):
            H_true = label[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            H_hat = outputs_lstm[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            Rate2[s] += get_zf_rate(H_hat, H_true, SNR) / N_it
            error = np.sum(np.abs(H_true - H_hat) ** 2)
            power = np.sum(np.abs(H_true) ** 2)
            NMSE2[s] += error / power / N_it

        # gru
        for s in range(args.pred_len + 1):
            H_true = label[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            H_hat = outputs_gru[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            Rate3[s] += get_zf_rate(H_hat, H_true, SNR) / N_it
            error = np.sum(np.abs(H_true - H_hat) ** 2)
            power = np.sum(np.abs(H_true) ** 2)
            NMSE3[s] += error / power / N_it

        # rnn
        for s in range(args.pred_len + 1):
            H_true = label[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            H_hat = outputs_rnn[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            Rate4[s] += get_zf_rate(H_hat, H_true, SNR) / N_it
            error = np.sum(np.abs(H_true - H_hat) ** 2)
            power = np.sum(np.abs(H_true) ** 2)
            NMSE4[s] += error / power / N_it

        # freq AR
        for s in range(args.pred_len + 1):
            H_true = label[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            H_hat = outputs_AR_freq[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            Rate5[s] += get_zf_rate(H_hat, H_true, SNR) / N_it
            error = np.sum(np.abs(H_true - H_hat) ** 2)
            power = np.sum(np.abs(H_true) ** 2)
            NMSE5[s] += error / power / N_it

        # delay AR
        for s in range(args.pred_len + 1):
            H_true = label[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            H_hat = outputs_AR_delay[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            Rate6[s] += get_zf_rate(H_hat, H_true, SNR) / N_it
            error = np.sum(np.abs(H_true - H_hat) ** 2)
            power = np.sum(np.abs(H_true) ** 2)
            NMSE6[s] += error / power / N_it

        # Previous
        for s in range(args.pred_len + 1):
            H_true = label[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            H_hat = data[:, -args.pred_len - 1, ...] if s > 0 else data[:, -args.pred_len - 1, ...]
            Rate7[s] += get_zf_rate(H_hat, H_true, SNR) / N_it
            error = np.sum(np.abs(H_true - H_hat) ** 2)
            power = np.sum(np.abs(H_true) ** 2)
            NMSE7[s] += error / power / N_it

        # hybrid
        for s in range(args.pred_len + 1):
            H_true = label[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            H_hat = outputs_hybrid[:, s - 1, :, :] if s > 0 else data[:, -args.pred_len - 1, ...]
            Rate_hybrid[s] += get_zf_rate(H_hat, H_true, SNR) / N_it
            error = np.sum(np.abs(H_true - H_hat) ** 2)
            power = np.sum(np.abs(H_true) ** 2)
            NMSE_hybrid[s] += error / power / N_it

        print('[{0}]/[{1}] nmse_informer:{2} nmse_lstm:{3} nmse_hybrid:{4}'.format(it + 1, N_it, nmse_informer, nmse_lstm, nmse_hybrid_val))

# plotting (include hybrid)
plt.figure()
x = np.array(list(range(data.shape[1])))
plt.plot(x, data[0, :, 0, 0], '--')
plt.plot(x[-args.pred_len:], outputs_informer[0, :, 0, 0], label='informer')
plt.plot(x[-args.pred_len:], outputs_hybrid[0, :, 0, 0], label='hybrid')
plt.legend()
plt.savefig('pred.png')
scio.savemat('./results/CR_data_' + str(args.v_max) + '.mat', {'data': data})
scio.savemat('./results/CR_transformer_' + str(args.v_max) + '.mat', {'data': outputs_informer})
scio.savemat('./results/CR_hybrid_' + str(args.v_max) + '.mat', {'data': outputs_hybrid})

plt.figure()
plt.plot(Rate0, '--')
plt.plot(Rate1)
plt.plot(Rate8)
plt.plot(Rate2)
plt.plot(Rate3)
plt.plot(Rate4)
plt.plot(Rate5)
plt.plot(Rate6)
plt.plot(Rate7)
plt.plot(Rate_hybrid)
plt.legend(['Ideal', 'Transformer_e2e', 'Transformer', 'LSTM', 'GRU', 'RNN', 'Freq AR', 'Delay AR', 'Previous', 'Hybrid'])
plt.xlabel('SRS (0.625 ms)')
plt.ylabel('Average rate (bit/s/Hz)')
plt.savefig('rate.png')

eps = 1e-10
plt.figure()
plt.plot(10 * np.log10(NMSE1 + eps))
plt.plot(10 * np.log10(NMSE8 + eps))
plt.plot(10 * np.log10(NMSE2 + eps))
plt.plot(10 * np.log10(NMSE3 + eps))
plt.plot(10 * np.log10(NMSE4 + eps))
plt.plot(10 * np.log10(NMSE5 + eps))
plt.plot(10 * np.log10(NMSE6 + eps))
plt.plot(10 * np.log10(NMSE7 + eps))
plt.plot(10 * np.log10(NMSE_hybrid + eps))
plt.legend(['Transformer_e2e', 'Transformer', 'LSTM', 'GRU', 'RNN', 'Freq AR', 'Delay AR', 'Previous', 'Hybrid'])
plt.xlabel('SRS (0.625 ms)')
plt.ylabel('NMSE (dB)')
plt.savefig('NMSE.png')

# save mats
scio.savemat('./results/Rate_Ideal_' + str(args.v_max) + '.mat', {'rate': Rate0})
scio.savemat('./results/Rate_transformer_' + str(args.v_max) + '.mat', {'rate': Rate8})
scio.savemat('./results/Rate_transformer_e2e_' + str(args.v_max) + '.mat', {'rate': Rate1})
scio.savemat('./results/Rate_LSTM_' + str(args.v_max) + '.mat', {'rate': Rate2})
scio.savemat('./results/Rate_GRU_' + str(args.v_max) + '.mat', {'rate': Rate3})
scio.savemat('./results/Rate_RNN_' + str(args.v_max) + '.mat', {'rate': Rate4})
scio.savemat('./results/Rate_PVEC_' + str(args.v_max) + '.mat', {'rate': Rate5})
scio.savemat('./results/Rate_PAD_' + str(args.v_max) + '.mat', {'rate': Rate6})
scio.savemat('./results/Rate_Previous_' + str(args.v_max) + '.mat', {'rate': Rate7})
scio.savemat('./results/Rate_Hybrid_' + str(args.v_max) + '.mat', {'rate': Rate_hybrid})

scio.savemat('./results/NMSE_Ideal_' + str(args.v_max) + '.mat', {'NMSE': NMSE0})
scio.savemat('./results/NMSE_transformer_' + str(args.v_max) + '.mat', {'NMSE': NMSE8})
scio.savemat('./results/NMSE_transformer_e2e_' + str(args.v_max) + '.mat', {'NMSE': NMSE1})
scio.savemat('./results/NMSE_LSTM_' + str(args.v_max) + '.mat', {'NMSE': NMSE2})
scio.savemat('./results/NMSE_GRU_' + str(args.v_max) + '.mat', {'NMSE': NMSE3})
scio.savemat('./results/NMSE_RNN_' + str(args.v_max) + '.mat', {'NMSE': NMSE4})
scio.savemat('./results/NMSE_PVEC_' + str(args.v_max) + '.mat', {'NMSE': NMSE5})
scio.savemat('./results/NMSE_PAD_' + str(args.v_max) + '.mat', {'NMSE': NMSE6})
scio.savemat('./results/NMSE_Previous_' + str(args.v_max) + '.mat', {'NMSE': NMSE7})
scio.savemat('./results/NMSE_Hybrid_' + str(args.v_max) + '.mat', {'NMSE': NMSE_hybrid})

print("Hybrid integration complete. Files saved: pred.png, rate.png, NMSE.png and .mat files in ./results/")
