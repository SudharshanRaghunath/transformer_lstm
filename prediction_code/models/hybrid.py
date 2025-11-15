"""
Hybrid Transformer + LSTM model for CSI prediction.

Place this file under `models/hybrid.py` in your repo and import as:

    from models.hybrid import HybridTransformerLSTM

Features:
- Transformer encoder over time dimension (captures long-range temporal patterns)
- LSTM decoder autoregressively predicts future frames
- Optional teacher forcing during training (forward supports `tgt` and `teacher_forcing_ratio`)
- `generate()` method for clean inference
- `save_checkpoint()` and `load_checkpoint()` helpers

Input / output shapes:
- enc_inp: [B, seq_len, enc_in] (float, real/imag flattened per time-slot)
- tgt (optional, for teacher forcing): [B, pred_len, enc_in]
- forward returns: [B, pred_len, enc_in]

Make sure the `enc_in` (feature dim) you pass here matches the rest of your code (default in repo is 16).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HybridTransformerLSTM(nn.Module):
    def __init__(self,
                 enc_in: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 trans_layers: int = 2,
                 lstm_hidden: int = 256,
                 lstm_layers: int = 1,
                 pred_len: int = 5,
                 dropout: float = 0.05,
                 max_pos_len: int = 512):
        """Constructor.

        Args:
            enc_in: input feature dimension per time-step (e.g. 16)
            d_model: transformer feature dimensionality
            nhead: number of attention heads
            trans_layers: number of transformer encoder layers
            lstm_hidden: hidden size of LSTM decoder
            lstm_layers: number of LSTM layers
            pred_len: number of frames to predict
            dropout: dropout in transformer layers
            max_pos_len: maximum supported time steps for positional embeddings
        """
        super().__init__()
        self.enc_in = enc_in
        self.d_model = d_model
        self.pred_len = pred_len

        # project input features to d_model
        self.input_proj = nn.Linear(enc_in, d_model)
        # optional layernorm for stability
        self.input_ln = nn.LayerNorm(d_model)

        # positional embeddings (learnable)
        self.pos_emb = nn.Parameter(torch.randn(1, max_pos_len, d_model))

        # transformer encoder (PyTorch native)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=d_model * 4,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)

        # LSTM decoder
        # We feed the decoder with d_model embeddings; it outputs lstm_hidden which we map back to enc_in
        self.lstm = nn.LSTM(input_size=d_model,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True)
        self.dec_proj = nn.Linear(lstm_hidden, enc_in)

        # to map predicted frame back to d_model for next-step input (feedback)
        self.feed_back_proj = nn.Linear(enc_in, d_model)

        # initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        # small init for stability
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, enc_inp: torch.Tensor, tgt: Optional[torch.Tensor] = None, teacher_forcing_ratio: float = 0.0):
        """
        Forward pass.

        Args:
            enc_inp: [B, seq_len, enc_in]
            tgt: optional ground-truth future sequence [B, pred_len, enc_in], used for teacher forcing
            teacher_forcing_ratio: float in [0,1], probability of using ground-truth at each decode step

        Returns:
            preds: [B, pred_len, enc_in]
        """
        b, seq_len, _ = enc_inp.shape
        device = enc_inp.device

        # encoder side
        x = self.input_proj(enc_inp)  # [B, seq_len, d_model]
        x = self.input_ln(x)
        # add positional (slice to seq_len)
        if seq_len <= self.pos_emb.size(1):
            x = x + self.pos_emb[:, :seq_len, :]
        else:
            # if seq longer than pos_emb we broadcast last positions (unlikely for your setting)
            pos = self.pos_emb[:, :self.pos_emb.size(1), :]
            pos = F.interpolate(pos.permute(0,2,1), seq_len, mode='linear', align_corners=False).permute(0,2,1)
            x = x + pos

        # transformer encoding
        enc_out = self.transformer(x)  # [B, seq_len, d_model]

        # decoder initialization: seed with last encoder output
        dec_input = enc_out[:, -1:, :]  # [B,1,d_model]

        # initialize LSTM hidden states (zeros)
        # shape for h0/c0: [num_layers, batch, hidden_size]
        num_layers = self.lstm.num_layers
        hidden = (torch.zeros(num_layers, b, self.lstm.hidden_size, device=device),
                  torch.zeros(num_layers, b, self.lstm.hidden_size, device=device))

        preds = []
        # if teacher forcing desired, make sure tgt is provided
        use_tf = (tgt is not None) and (teacher_forcing_ratio > 0.0)

        # decode pred_len steps
        for t in range(self.pred_len):
            # LSTM expects input [B, seq_len=1, d_model]
            out, hidden = self.lstm(dec_input, hidden)  # out: [B,1,lstm_hidden]
            frame = self.dec_proj(out.squeeze(1))  # [B, enc_in]
            preds.append(frame.unsqueeze(1))

            if use_tf and torch.rand(1).item() < teacher_forcing_ratio:
                # use ground-truth next frame as next input
                next_in = tgt[:, t:t+1, :]
                # project back to d_model
                dec_input = self.feed_back_proj(next_in)
            else:
                # use model prediction as next input
                dec_input = self.feed_back_proj(frame.unsqueeze(1))

        preds = torch.cat(preds, dim=1)  # [B, pred_len, enc_in]
        return preds

    def generate(self, enc_inp: torch.Tensor, pred_len: Optional[int] = None) -> torch.Tensor:
        """
        Strict inference mode without teacher forcing.
        Returns predictions as [B, pred_len, enc_in].
        """
        if pred_len is None:
            pred_len = self.pred_len
        # call forward with no tgt and tf=0
        old_pred = self.pred_len
        self.pred_len = pred_len
        with torch.no_grad():
            preds = self.forward(enc_inp, tgt=None, teacher_forcing_ratio=0.0)
        self.pred_len = old_pred
        return preds

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, epoch: Optional[int] = None):
        """Save checkpoint dict to given path."""
        ckpt = {
            'state_dict': self.state_dict()
        }
        if optimizer is not None:
            ckpt['optimizer'] = optimizer.state_dict()
        if epoch is not None:
            ckpt['epoch'] = epoch
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, map_location: Optional[str] = None, strict: bool = True):
        """Load checkpoint from path. Accepts full dict or just state_dict."""
        loc = map_location if map_location is not None else None
        raw = torch.load(path, map_location=loc)
        state = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw
        self.load_state_dict(state, strict=strict)


# quick smoke test when run as script
if __name__ == '__main__':
    # sanity-check shapes
    B, seq_len, enc_in = 2, 25, 16
    model = HybridTransformerLSTM(enc_in=enc_in, d_model=64, nhead=4, trans_layers=2, lstm_hidden=128, lstm_layers=1, pred_len=5)
    x = torch.randn(B, seq_len, enc_in)
    tgt = torch.randn(B, 5, enc_in)
    out = model(x, tgt=tgt, teacher_forcing_ratio=0.5)
    print('out shape', out.shape)  # should be [B, 5, enc_in]
