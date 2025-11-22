"""
Robust Hybrid Transformer + LSTM model for CSI prediction.

Drop into: prediction_code/models/hybrid.py

Key behavior:
- Accepts either flattened input [B, seq_len, enc_in] OR raw channel tensor
  [T, M, Nr, Nt] (or [M, T, Nr, Nt]) and uses data.LoadBatch to convert.
- If converted feature-dimension != expected enc_in, create a learnable
  input_adapter (Linear) to map incoming dimension -> enc_in (registered once).
- Transformer encoder followed by LSTM autoregressive decoder.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# lazy import of LoadBatch (used to convert raw channel tensors to flattened input)
try:
    from data import LoadBatch
except Exception:
    LoadBatch = None  # will raise later if conversion is needed but LoadBatch not found


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
        super().__init__()
        self.enc_in = enc_in
        self.d_model = d_model
        self.pred_len = pred_len

        # project input features to d_model
        self.input_proj = nn.Linear(enc_in, d_model)
        self.input_ln = nn.LayerNorm(d_model)

        # position embeddings
        self.pos_emb = nn.Parameter(torch.randn(1, max_pos_len, d_model))

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=d_model * 4,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)

        # LSTM decoder
        self.lstm = nn.LSTM(input_size=d_model,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True)
        self.dec_proj = nn.Linear(lstm_hidden, enc_in)
        self.feed_back_proj = nn.Linear(enc_in, d_model)

        # adapter (created on-demand if input feature size != enc_in)
        self.input_adapter: Optional[nn.Module] = None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # --------------------- helpers to accept raw inputs ---------------------
    def _convert_raw_to_flat(self, enc_inp, device=None):
        """
        Convert raw channel tensors (numpy or torch) into flattened [M, seq_len, feat]
        using project's LoadBatch function.

        Returns a torch.Tensor shaped [B, seq_len, feat_in] where B == M (treated as batch).
        """
        if LoadBatch is None:
            raise RuntimeError("LoadBatch not importable from data.py — ensure module is on PYTHONPATH.")

        was_torch = isinstance(enc_inp, torch.Tensor)
        if was_torch:
            arr = enc_inp.detach().cpu().numpy()
        else:
            arr = np.array(enc_inp)

        # Try direct pass; if it fails try transposed version (many code variants exist)
        last_err = None
        try:
            flat = LoadBatch(arr)  # expected to return [M, seq_len, feat]
        except Exception as e:
            last_err = e
            try:
                arr_t = arr.transpose([1, 0, 2, 3])
                flat = LoadBatch(arr_t)
            except Exception as e2:
                raise RuntimeError(f"LoadBatch conversion failed. Tried both orientations: {e}; {e2}")

        # Convert to torch
        if isinstance(flat, np.ndarray):
            flat_t = torch.from_numpy(flat).float()
        else:
            flat_t = flat.float()

        if device is not None:
            flat_t = flat_t.to(device)

        return flat_t

    # --------------------- forward / generate ---------------------
    def forward(self, enc_inp: torch.Tensor, tgt: Optional[torch.Tensor] = None, teacher_forcing_ratio: float = 0.0):
        """
        enc_inp: either
            - torch.Tensor [B, seq_len, enc_in] (flattened) OR
            - raw array/tensor [T, M, Nr, Nt] or [M, T, Nr, Nt] (numpy or torch)
        """
        device = None
        # 1) Convert raw channel tensor into flattened form if needed
        if isinstance(enc_inp, torch.Tensor):
            device = enc_inp.device
            if enc_inp.dim() == 4 or enc_inp.dim() == 5:
                # likely raw: convert to flattened using LoadBatch
                enc_inp = self._convert_raw_to_flat(enc_inp, device=device)
        else:
            # numpy or other object
            tmp = np.array(enc_inp)
            if tmp.ndim == 4 or tmp.ndim == 5:
                enc_inp = self._convert_raw_to_flat(tmp, device=device)
            else:
                enc_inp = torch.from_numpy(tmp).float()

        # enforce tensor and dims
        if not isinstance(enc_inp, torch.Tensor):
            enc_inp = torch.from_numpy(np.array(enc_inp)).float()
        if enc_inp.dim() == 2:
            enc_inp = enc_inp.unsqueeze(0)
        if enc_inp.dim() != 3:
            raise ValueError(
                f"Model.forward: expected enc_inp with 3 dims [B, seq_len, feat], got shape {tuple(enc_inp.shape)}")

        b, seq_len, feat = enc_inp.shape
        device = enc_inp.device

        # 2) If feature-dim mismatch, create / use input_adapter to map feat -> self.enc_in
        if feat != self.enc_in:
            # create adapter on first use (register as module so it's saved & trained)
            if (self.input_adapter is None) or (getattr(self.input_adapter, 'in_features', None) != feat):
                print(f"[Hybrid] creating input_adapter to map {feat} -> {self.enc_in}")
                adapter = nn.Linear(feat, self.enc_in)
                # initialize adapter
                nn.init.xavier_uniform_(adapter.weight)
                adapter = adapter.to(device)
                # register adapter as submodule so optimizer will see it
                self.add_module('input_adapter_auto', adapter)
                self.input_adapter = adapter
            # apply adapter
            enc_inp = self.input_adapter(enc_inp)  # now [B, seq_len, self.enc_in]
            feat = self.enc_in

        # 3) normal pipeline: project -> transformer -> decode
        x = self.input_proj(enc_inp)  # [B, seq_len, d_model]
        x = self.input_ln(x)
        if seq_len <= self.pos_emb.size(1):
            x = x + self.pos_emb[:, :seq_len, :].to(device)
        else:
            pos = self.pos_emb[:, :self.pos_emb.size(1), :].to(device)
            pos = F.interpolate(pos.permute(0, 2, 1), seq_len, mode='linear', align_corners=False).permute(0, 2, 1)
            x = x + pos

        enc_out = self.transformer(x)  # [B, seq_len, d_model]

        dec_input = enc_out[:, -1:, :]  # [B,1,d_model]
        num_layers = self.lstm.num_layers
        hidden = (torch.zeros(num_layers, b, self.lstm.hidden_size, device=device),
                  torch.zeros(num_layers, b, self.lstm.hidden_size, device=device))

        preds = []
        use_tf = (tgt is not None) and (teacher_forcing_ratio > 0.0)

        for t in range(self.pred_len):
            out, hidden = self.lstm(dec_input, hidden)
            frame = self.dec_proj(out.squeeze(1))  # [B, enc_in]
            preds.append(frame.unsqueeze(1))

            if use_tf and torch.rand(1).item() < teacher_forcing_ratio:
                next_in = tgt[:, t:t + 1, :]
                dec_input = self.feed_back_proj(next_in)
            else:
                dec_input = self.feed_back_proj(frame.unsqueeze(1))

        preds = torch.cat(preds, dim=1)  # [B, pred_len, enc_in]
        return preds

    def generate(self, enc_inp: torch.Tensor, pred_len: Optional[int] = None) -> torch.Tensor:
        if pred_len is None:
            pred_len = self.pred_len
        old = self.pred_len
        self.pred_len = pred_len
        with torch.no_grad():
            preds = self.forward(enc_inp, tgt=None, teacher_forcing_ratio=0.0)
        self.pred_len = old
        return preds

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                        epoch: Optional[int] = None):
        ckpt = {'state_dict': self.state_dict()}
        if optimizer is not None:
            ckpt['optimizer'] = optimizer.state_dict()
        if epoch is not None:
            ckpt['epoch'] = epoch
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, map_location: Optional[str] = None, strict: bool = True):
        loc = map_location if map_location is not None else None
        raw = torch.load(path, map_location=loc)
        state = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw
        self.load_state_dict(state, strict=strict)


# quick smoke test
if __name__ == '__main__':
    B, seq_len, enc_in = 2, 25, 16
    model = HybridTransformerLSTM(enc_in=enc_in, d_model=64, nhead=4, trans_layers=2, lstm_hidden=128, lstm_layers=1,
                                  pred_len=5)
    x = torch.randn(B, seq_len, enc_in)
    tgt = torch.randn(B, 5, enc_in)
    out = model(x, tgt=tgt, teacher_forcing_ratio=0.5)
    print('out shape', out.shape)
