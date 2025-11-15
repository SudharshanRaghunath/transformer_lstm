import torch
import torch.nn as nn

from encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from decoder import Decoder, DecoderLayer
from attn import FullAttention, ProbAttention, AttentionLayer
from embed import DataEmbedding


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', activation='gelu',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class InformerStack_e2e(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', activation='gelu',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        super(InformerStack_e2e, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        stacks = list(range(e_layers, 2, -1))  # you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in stacks]
        self.encoder = EncoderStack(encoders)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', activation='gelu',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        stacks = list(range(e_layers, 2, -1))  # you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in stacks]
        self.encoder = EncoderStack(encoders)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class RNNUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):
        super(RNNUnit, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        # self.out = nn.Linear(hidden_size, features)

    def forward(self, x, prev_hidden):
        # x expected shape: [L, B, F]
        L, B, F = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output)
        output = output.reshape(L, B, -1)
        output, cur_hidden = self.rnn(output, prev_hidden)
        output = output.reshape(L * B, -1)
        output = self.decoder(output)
        output = output.reshape(L, B, -1)

        return output, cur_hidden


class RNN(nn.Module):
    """
    Simple wrapper for RNNUnit to provide train/test APIs
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):

        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.features = features
        self.model = RNNUnit(features, input_size, hidden_size, num_layers=self.num_layers)

    def train_data(self, x, device):

        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device)
        outputs = []
        for idx in range(seq_len):
            output, prev_hidden = self.model(x[:, idx:idx + 1, ...].permute(1, 0, 2).contiguous(), prev_hidden)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()

        return outputs

    def test_data(self, x, pred_len, device):
        """
        Improved and robust autoregressive generation.
        Input:
          x: [B, seq_len, features] (torch tensor)
          pred_len: number of future steps to generate
        Output:
          outputs: [B, pred_len, features]
        Behavior:
          - Process the input sequence step-by-step to update hidden state
          - Use the model's last output as the seed and autoregressively generate pred_len future frames
          - Returns only the predicted future frames (not reconstruction of last input)
        """

        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device)
        outputs_list = []

        last_output = None
        # 1) feed observed inputs
        for idx in range(seq_len):
            inp_step = x[:, idx:idx + 1, ...].permute(1, 0, 2).contiguous()  # [1, B, F]
            out_step, prev_hidden = self.model(inp_step, prev_hidden)
            last_output = out_step

        # 2) autoregressive generation for pred_len steps
        cur = last_output
        for _ in range(pred_len):
            out_step, prev_hidden = self.model(cur, prev_hidden)
            outputs_list.append(out_step)
            cur = out_step

        # concatenate outputs -> [pred_len, B, F] then -> [B, pred_len, F]
        outputs = torch.cat(outputs_list, dim=0).permute(1, 0, 2).contiguous()
        return outputs


class GRUUnit(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):
        super(GRUUnit, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))

    def forward(self, x, prev_hidden):
        # x: [L, B, F]
        L, B, F = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output)
        output = output.reshape(L, B, -1)
        output, cur_hidden = self.gru(output, prev_hidden)
        output = output.reshape(L * B, -1)
        output = self.decoder(output)
        output = output.reshape(L, B, -1)

        return output, cur_hidden


class GRU(nn.Module):
    """
    Wrapper that exposes train_data/test_data APIs for GRUUnit
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):

        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.features = features
        self.model = GRUUnit(features, input_size, hidden_size, num_layers=self.num_layers)

    def train_data(self, x, device):

        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device)
        outputs = []
        for idx in range(seq_len):
            output, prev_hidden = self.model(x[:, idx:idx + 1, ...].permute(1, 0, 2).contiguous(), prev_hidden)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()

        return outputs

    def test_data(self, x, pred_len, device):
        """
        Robust autoregressive generation for GRU
        Returns: [B, pred_len, features]
        """
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device)
        outputs_list = []

        last_output = None
        # feed observed inputs
        for idx in range(seq_len):
            inp_step = x[:, idx:idx + 1, ...].permute(1, 0, 2).contiguous()
            out_step, prev_hidden = self.model(inp_step, prev_hidden)
            last_output = out_step

        # autoregress
        cur = last_output
        for _ in range(pred_len):
            out_step, prev_hidden = self.model(cur, prev_hidden)
            outputs_list.append(out_step)
            cur = out_step

        outputs = torch.cat(outputs_list, dim=0).permute(1, 0, 2).contiguous()
        return outputs


class LSTMUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):
        super(LSTMUnit, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))

    def forward(self, x, prev_hidden, prev_cell):
        # x: [L, B, F]
        L, B, F = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output)
        output = output.reshape(L, B, -1)
        output, (cur_hidden, cur_cell) = self.lstm(output, (prev_hidden, prev_cell))
        output = output.reshape(L * B, -1)
        output = self.decoder(output)
        output = output.reshape(L, B, -1)

        return output, cur_hidden, cur_cell


class LSTM(nn.Module):
    """
    Wrapper exposing train/test APIs for LSTMUnit
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):

        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.features = features
        self.model = LSTMUnit(features, input_size, hidden_size, num_layers=self.num_layers)

    def train_data(self, x, device):

        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device)
        prev_cell = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device)
        outputs = []
        for idx in range(seq_len):
            output, prev_hidden, prev_cell = self.model(x[:, idx:idx + 1, ...].permute(1, 0, 2).contiguous(),
                                                        prev_hidden, prev_cell)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()

        return outputs

    def test_data(self, x, pred_len, device):
        """
        Robust autoregressive generation for LSTM
        Input:
          x: [B, seq_len, features]
          pred_len: number of future frames
        Output:
          outputs: [B, pred_len, features]
        """
        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device)
        prev_cell = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device)
        outputs_list = []

        last_output = None
        # feed observed inputs
        for idx in range(seq_len):
            inp_step = x[:, idx:idx + 1, ...].permute(1, 0, 2).contiguous()
            out_step, prev_hidden, prev_cell = self.model(inp_step, prev_hidden, prev_cell)
            last_output = out_step

        # autoregress
        cur = last_output
        for _ in range(pred_len):
            out_step, prev_hidden, prev_cell = self.model(cur, prev_hidden, prev_cell)
            outputs_list.append(out_step)
            cur = out_step

        outputs = torch.cat(outputs_list, dim=0).permute(1, 0, 2).contiguous()
        return outputs
