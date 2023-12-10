import random

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from alaapNet.tools.utils import Util
from alaapNet.models.modules import PositionalEncoding
from alaapNet.models.conv_seq2seq import ConvSeq2Seq
from alaapNet.models.gamakanet import GamakaNet


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Two layers are used so a non-linear act function can be placed
        # in between
        self.attn = nn.Linear(2 * hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden_final_layer, encoder_outputs):
        # decoder_hidden_final_layer: (batch size, hidden size)
        # encoder_outputs: (batch size, input seq len, hidden size)

        # Repeat decoder hidden state input seq len times
        hidden = decoder_hidden_final_layer.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1)

        # Compare decoder hidden state with each encoder output using a learnable tanh layer
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # Then compress into single values for each comparison (energy)
        attention = self.v(energy).squeeze(2)

        # Then softmax so the weightings add up to 1
        weightings = F.softmax(attention, dim=1)

        # weightings: (batch size, input seq len)
        return weightings


class Encoder(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.GRU(input_size=feature_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.constant_(self.lstm.bias_ih_l0, 0)
        nn.init.constant_(self.lstm.bias_hh_l0, 0)

    def forward(self, x: torch.Tensor):
        out, hidden = self.lstm(x)
        return out, hidden


# Decoder superclass whose forward is called by Seq2Seq but other methods filled out by subclasses
class DecoderBase(nn.Module):
    def __init__(self, dec_target_size, target_indices):
        super().__init__()
        self.target_indices = target_indices
        self.target_size = dec_target_size

    # Have to run one step at a time unlike with the encoder since sometimes not teacher forcing
    def run_single_recurrent_step(self, inputs, hidden, enc_outputs):
        raise NotImplementedError()

    def forward(self, target, hidden, enc_outputs, input_seq_len, teacher_force_prob=None):
        # target: (batch size, output seq length, num dec features)
        # hidden: (num gru layers, batch size, hidden dim), ie the last hidden state
        # enc_outputs: (batch size, input seq len, hidden size)

        batch_size, dec_output_seq_length, ft_size = target.shape

        # Store decoder outputs
        # outputs: (batch size, output seq len, num targets, num dist params)
        outputs = torch.zeros(batch_size, dec_output_seq_length, ft_size,
                              dtype=torch.float, device=target.device)

        # curr_input: (batch size, 1, num dec features)
        curr_input = target[:, 0:1, :]

        for t in range(dec_output_seq_length):
            # dec_output: (batch size, 1, num targets, num dist params)
            # hidden: (num gru layers, batch size, hidden size)
            dec_output, hidden = self.run_single_recurrent_step(curr_input, hidden, enc_outputs)
            # Save prediction
            dec_output = F.relu(dec_output)
            outputs[:, t:t + 1, :] = dec_output

            # If teacher forcing, use target from this timestep as next input o.w. use prediction
            teacher_force = random.random() < teacher_force_prob if teacher_force_prob is not None else False

            curr_input = target[:, t:t + 1, :].clone()
            if not teacher_force:
                curr_input[:, :, self.target_indices] = dec_output

        return outputs

    @staticmethod
    def layer_init(layer, w_scale=1.0):
        nn.init.kaiming_uniform_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0.)
        return layer


class DecoderVanilla(DecoderBase):
    def __init__(self, dec_feature_size, dec_target_size, hidden_size,
                 num_gru_layers, target_indices, dropout):
        super().__init__(dec_target_size, target_indices)
        self.gru = nn.GRU(dec_feature_size, hidden_size, num_gru_layers, batch_first=True, dropout=dropout)
        self.out = DecoderBase.layer_init(nn.Linear(hidden_size + dec_feature_size, dec_target_size))

    def run_single_recurrent_step(self, inputs, hidden, enc_outputs):
        # inputs: (batch size, 1, num dec features)
        # hidden: (num gru layers, batch size, hidden size)

        output, hidden = self.gru(inputs, hidden)
        output = self.out(torch.cat((output, inputs), dim=2))
        output = output.view(output.shape[0], output.shape[1], self.target_size)
        # output: (batch size, 1, num targets, num dist params)
        # hidden: (num gru layers, batch size, hidden size)
        return output, hidden


class DecoderWithAttention(DecoderBase):
    def __init__(self, dec_feature_size, dec_target_size, hidden_size,
                 num_gru_layers, target_indices, dropout):
        super().__init__(dec_target_size, target_indices)
        self.attention_model = Attention(hidden_size)
        # GRU takes previous timestep target and weighted sum of encoder hidden states
        self.gru = nn.GRU(dec_feature_size + hidden_size, hidden_size, num_gru_layers, batch_first=True,
                          dropout=dropout)
        # Output layer takes decoder hidden state output, weighted sum and decoder input
        # NOTE: Feeding decoder input into the output layer essentially acts as a skip connection
        self.out = DecoderBase.layer_init(nn.Linear(hidden_size + hidden_size + dec_feature_size,
                                                    dec_target_size))

    def run_single_recurrent_step(self, inputs, hidden, enc_outputs):
        # inputs: (batch size, 1, num dec features)
        # hidden: (num gru layers, batch size, hidden size)
        # enc_outputs: (batch size, input seq len, hidden size)

        # Get attention weightings
        # weightings: (batch size, input seq len)
        weightings = self.attention_model(hidden[-1], enc_outputs)

        # Then compute weighted sum
        # weighted_sum: (batch size, 1, hidden size)
        weighted_sum = torch.bmm(weightings.unsqueeze(1), enc_outputs)

        # Then input into GRU
        # gru inputs: (batch size, 1, num dec features + hidden size)
        # output: (batch size, 1, hidden size)
        output, hidden = self.gru(torch.cat((inputs, weighted_sum), dim=2), hidden)

        # Get prediction
        # out input: (batch size, 1, hidden size + hidden size + num targets)
        output = self.out(torch.cat((output, weighted_sum, inputs), dim=2))
        output = output.view(output.shape[0], output.shape[1], self.target_size)

        # output: (batch size, 1, num targets, num dist params)
        # hidden: (num gru layers, batch size, hidden size)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, dropout_p,
                 target_indices):
        super().__init__()
        self.encoder = Encoder(feature_size, hidden_size, num_layers)
        self.decoder = DecoderWithAttention(feature_size, feature_size,
                                            hidden_size, num_layers, target_indices, dropout_p)

    def forward(self, enc_inputs, dec_inputs, teacher_force_prob=None):
        # enc_inputs: (batch size, input seq length, num enc features)
        # dec_inputs: (batch size, output seq length, num dec features)

        # enc_outputs: (batch size, input seq len, hidden size)
        # hidden: (num gru layers, batch size, hidden dim), ie the last hidden state
        enc_outputs, hidden = self.encoder(enc_inputs)

        # outputs: (batch size, output seq len, num targets, num dist params)
        outputs = self.decoder(dec_inputs, hidden, enc_outputs, enc_inputs.shape[1], teacher_force_prob)

        return outputs


class ConvNet(nn.Module):
    def __init__(self, out_seq_len, activation: nn.Module):
        super().__init__()

        self.conv = nn.ModuleList()

        self.conv.append(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=2))
        self.conv.append(nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=2))
        self.conv.append(nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2))
        self.conv.append(nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1))
        self.conv.append(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1))

        self.attn_fc_id = [1, 0, 1, 0, 0]

        assert len(self.conv) == len(self.attn_fc_id)

        self.attn_fc = nn.ModuleList()
        self.attn_fc.append(nn.Linear(in_features=8 * 48, out_features=128))
        self.attn_fc.append(nn.Linear(in_features=32 * 9, out_features=128))

        self.fc = nn.Linear(in_features=128, out_features=out_seq_len)
        self.act = activation

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        attn = []
        k = 0
        for i in range(len(self.conv)):
            x = self.conv[i](x)
            x = self.act(x)
            if self.attn_fc_id[i] != 0:
                temp = x.reshape(batch_size, -1)
                temp = self.attn_fc[k](temp)
                attn.append(temp)
                k += 1

        x = x.reshape(x.shape[0], -1)
        temp = torch.stack(attn, dim=-1)
        temp = torch.sum(temp, dim=-1)
        x = x + temp
        x = self.fc(x)
        x = self.act(x)
        return x.reshape(x.shape[0], -1, 1)

# class DecoderCell(nn.Module):
#     def __init__(self, feature_size, hidden_size, num_layers, dropout_p):
#         super().__init__()
#         self.lstm = nn.GRU(input_size=feature_size,
#                            hidden_size=hidden_size,
#                            num_layers=num_layers,
#                            batch_first=True)
#
#         self.fc = nn.Linear(hidden_size, feature_size)
#         self.activation = nn.Sigmoid()
#         self.dropout = nn.Dropout(dropout_p)
#
#         nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
#         nn.init.orthogonal_(self.lstm.weight_hh_l0)
#         nn.init.constant_(self.lstm.bias_ih_l0, 0)
#         nn.init.constant_(self.lstm.bias_hh_l0, 0)
#
#         nn.init.xavier_uniform_(self.fc.weight)
#         nn.init.orthogonal_(self.fc.weight)
#         nn.init.constant_(self.fc.bias, 0)
#
#     def forward(self, x: torch.Tensor, hidden_state: torch.Tensor, encoder_outputs=None):
#         x, hidden = self.lstm(x, hidden_state)
#         x = torch.clamp(x, min=0, max=5.0)
#         x = self.fc(x)
#         return self.activation(x), hidden
#
#
# class Decoder(nn.Module):
#     def __init__(self, feature_size, hidden_size, num_layers, dropout_p):
#         super().__init__()
#         self.decoder_cell = DecoderCell(feature_size, hidden_size, num_layers, dropout_p)
#
#     def forward(self,
#                 encoder_outputs,
#                 hidden,
#                 target=None,
#                 tokens: List[torch.Tensor] or None = None,
#                 teacher_force_probability=0.5,
#                 max_length=2000):
#
#         batch_size = len(encoder_outputs)
#
#         if target is not None:  # Training
#             batch_size, seq_len, _ = target.shape
#             outputs = torch.zeros((batch_size, seq_len, self.feature_size), device=encoder_outputs.device)
#             # start token
#             x = target[:, 0].unsqueeze(1)
#             outputs[:, 0] = target[:, 0]
#
#             for t in range(1, seq_len):
#                 out, hidden = self.decoder_cell(x, hidden)
#                 outputs[:, t] = out[:, 0]
#                 x = target[:, t].unsqueeze(1) if random.random() < teacher_force_probability else out.detach()
#
#             return outputs
#
#         elif tokens is not None:  # inference
#             outputs = torch.zeros((batch_size, max_length, self.feature_size), device=encoder_outputs.device)
#             x = torch.tile(tokens[0], dims=(batch_size,)).reshape(batch_size, -1, self.feature_size)
#             outputs[:, 0] = x[:, 0]
#             for t in range(1, max_length):
#                 x, hidden = self.decoder_cell(x, hidden)
#                 outputs[:, t] = x[:, 0]
#
#             return outputs
#
#
# class AttnDecoderCell(nn.Module):
#     def __init__(self, feature_size, hidden_size, num_layers, bidirectional_encoder, dropout_p):
#         super().__init__()
#         self.attention = BahdanauAttention(hidden_size, num_layers, bidirectional_encoder)
#         self.decoder_cell = DecoderCell(feature_size, hidden_size, num_layers, dropout_p)
#
#     def forward(self,
#                 x: torch.Tensor,
#                 hidden: torch.Tensor,
#                 encoder_outputs: torch.Tensor):
#         query = hidden.permute(1, 0, 2)
#         cxt, attn_wts = self.attention(query, encoder_outputs)
#         x = torch.cat((x, cxt), dim=2)
#         out, hidden = self.decoder_cell(x, hidden)
#         return out, hidden, attn_wts
#
#
# class AttnDecoder(nn.Module):
#     def __init__(self, feature_size, hidden_size, num_layers, bidirectional_encoder, dropout_p):
#         super().__init__()
#         self.decoder_cell = AttnDecoderCell(feature_size, hidden_size, num_layers, bidirectional_encoder, dropout_p)
#         self.feature_size = feature_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.dropout_p = dropout_p
#
#     def forward(self,
#                 encoder_outputs: torch.Tensor,
#                 hidden: torch.Tensor,
#                 target: torch.Tensor,
#                 tokens=None,
#                 teacher_force_probability=0.5, max_length=2000):
#         attentions = []
#         batch_size, seq_len, _ = target.shape
#         outputs = torch.zeros((batch_size, seq_len, self.feature_size), device=encoder_outputs.device)
#         # start token
#         x = target[:, 0].unsqueeze(1)
#         outputs[:, 0] = target[:, 0]
#
#         print(x.shape, hidden.shape, encoder_outputs.shape)
#         for t in range(1, seq_len):
#             out, hidden, attn_wts = self.decoder_cell(x, hidden, encoder_outputs)
#             outputs[:, t] = out[:, 0]
#             x = target[:, t].unsqueeze(1) if random.random() < teacher_force_probability else out.detach()
#             attentions.append(attn_wts)
#
#         attentions = torch.cat(attentions, dim=1)
#         return outputs, hidden, attentions
#
#
# class Seq2Seq(nn.Module):
#     def __init__(self, feature_size, hidden_size, num_layers, bidirectional_encoder):
#         super().__init__()
#         self.encoder = Encoder(feature_size=feature_size,
#                                hidden_size=hidden_size,
#                                num_layers=num_layers,
#                                bidirectional=bidirectional_encoder)
#
#         self.D = 2 if bidirectional_encoder else 1
#         self.hidden_linear = nn.Linear(num_layers * self.D * hidden_size, num_layers * hidden_size)
#
#         # self.decoder = Decoder(feature_size=feature_size,
#         #                        hidden_size=hidden_size,
#         #                        num_layers=num_layers,
#         #                        dropout_p=0.25)
#
#         self.decoder = AttnDecoder(feature_size=feature_size,
#                                    hidden_size=hidden_size,
#                                    num_layers=num_layers,
#                                    bidirectional_encoder=bidirectional_encoder,
#                                    dropout_p=0.25)
#
#         self.feature_size = feature_size
#         self.num_layers = num_layers
#         self.bi = bidirectional_encoder
#
#     def forward(self, x: torch.Tensor,
#                 tokens: List[torch.Tensor] or None = None,
#                 target: torch.Tensor or None = None,
#                 teacher_force_probability=0.5,
#                 max_len=100):
#         assert (tokens is None) ^ (target is None), "Provide either tokens or target"
#         out, hidden = self.encoder(x)
#
#         batch_size = len(x)
#
#         if self.bi:
#             hidden = self.hidden_linear(hidden.permute(1, 0, 2).reshape(batch_size, -1))
#             hidden = hidden.reshape(self.num_layers, batch_size, -1)
#
#         return self.decoder(encoder_outputs=out,
#                             hidden=hidden,
#                             target=target,
#                             tokens=tokens,
#                             teacher_force_probability=teacher_force_probability,
#                             max_length=max_len)


class Seq2SeqComponent(nn.Module):
    def __init__(self, out_seq_len, alpha=0.9, kernel_size=3):
        super().__init__()
        # self.trend_model = Seq2Seq(feature_size=feature_size,
        #                            hidden_size=hidden_size,
        #                            num_layers=num_layers, dropout_p=0, target_indices=[0])

        # self.trend_model = GamakaNet(input_size=(feature_size, 1), latent_size=hidden_size)
        # self.seasonal_model = GamakaNet(input_size=(feature_size, 1), latent_size=hidden_size)
        # self.seasonal_model = Seq2Seq(feature_size=feature_size,
        #                               hidden_size=hidden_size,
        #                               num_layers=num_layers, dropout_p=0, target_indices=[0])

        self.trend_model = ConvNet(out_seq_len=out_seq_len, activation=nn.ReLU())
        self.seasonal_model = ConvNet(out_seq_len=out_seq_len, activation=nn.Tanh())

        # self.pool = nn.AvgPool1d(kernel_size=kernel_size, padding=kernel_size // 2)
        # self.seasonal_model = ConvSeq2Seq(feature_size=feature_size,
        #                                hidden_dim=hidden_size,
        #                                num_layers=num_layers,
        #                                kernel_size=kernel_size,
        #                                tgt_pad_idx=0,
        #                                dropout=0,
        #                                max_len=5000)
        self.alpha = alpha

    def forward(self, x: torch.Tensor):
        trend = Util.zero_lpf(x, self.alpha)
        # seasonal = x - trend
        trend_pred = self.trend_model(x)
        seasonal = self.seasonal_model(trend)
        return trend_pred + seasonal


class TransformerModel(nn.Module):
    def __init__(self, feature_size, d_model, n_heads, num_layers, dim_feedforward=2048, pos_enc_dropout=0.1):
        super().__init__()
        self.encoder_input_layer = nn.Linear(in_features=feature_size, out_features=d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=pos_enc_dropout)

        self.decoder_input_layer = nn.Linear(in_features=feature_size, out_features=d_model)
        self.linear_map = nn.Linear(in_features=d_model, out_features=feature_size)

        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=n_heads,
                                          batch_first=True,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dim_feedforward=dim_feedforward)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                src_pad_mask: torch.Tensor = None,
                tgt_pad_mask: torch.Tensor = None):
        src = self.encoder_input_layer(src)
        src = self.pos_encoding(src)

        tgt = self.decoder_input_layer(tgt)
        tgt = self.pos_encoding(tgt)
        out = self.transformer(src, tgt, tgt_mask, src_pad_mask, tgt_pad_mask)
        decoder_out = self.linear_map(out)

        return decoder_out

    @staticmethod
    def get_tgt_mask(size: int):
        mask = torch.tril(torch.ones(size, size) == 1).float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        return mask

    @staticmethod
    def create_pad_mask(x: torch.Tensor, pad_token: int):
        return x == pad_token
