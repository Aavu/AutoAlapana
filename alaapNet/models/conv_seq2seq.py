import torch
from torch import nn
import torch.nn.functional as F
from alaapNet.models.modules import PositionalEncoding


class Encoder(nn.Module):
    def __init__(self,
                 feature_size,
                 hidden_dim,
                 num_layers,
                 kernel_size,
                 dropout,
                 max_len):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))

        self.emb2hid = nn.Linear(feature_size, hidden_dim)
        self.hid2emb = nn.Linear(hidden_dim, feature_size)

        self.conv = nn.ModuleList([nn.Conv1d(in_channels=hidden_dim,
                                             out_channels=2 * hidden_dim,
                                             kernel_size=kernel_size,
                                             padding=(kernel_size - 1) // 2)
                                   for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = PositionalEncoding(d_model=feature_size, dropout=dropout, max_len=max_len)

    def forward(self, x: torch.Tensor):
        combined = self.dropout(self.pos_encoding(x) + x)
        x = self.emb2hid(combined)
        x = x.permute(0, 2, 1)

        self.scale = self.scale.to(x.device)
        for i, conv in enumerate(self.conv):
            y = conv(self.dropout(x))
            y = F.glu(y, dim=1)
            x = (y + x) * self.scale

        x = self.hid2emb(x.permute(0, 2, 1))
        combined = (x + combined) * self.scale
        return x, combined


class Decoder(nn.Module):
    def __init__(self,
                 feature_size,
                 hidden_dim,
                 num_layers,
                 kernel_size,
                 tgt_pad_idx,
                 dropout,
                 max_len):
        super().__init__()
        self.kernel_size = kernel_size
        self.tgt_pad_idx = tgt_pad_idx
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))

        self.emb2hid = nn.Linear(feature_size, hidden_dim)
        self.hid2emb = nn.Linear(hidden_dim, feature_size)

        self.attn_hid2emb = nn.Linear(hidden_dim, feature_size)
        self.attn_emb2hid = nn.Linear(feature_size, hidden_dim)

        self.conv = nn.ModuleList([nn.Conv1d(in_channels=hidden_dim,
                                             out_channels=2 * hidden_dim,
                                             kernel_size=kernel_size)
                                   for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = PositionalEncoding(d_model=feature_size, dropout=dropout, max_len=max_len)

    def forward(self, tgt: torch.Tensor, enc_conv: torch.Tensor, enc_comb: torch.Tensor):
        batch_size, tgt_len, ft_size = tgt.shape
        combined = self.dropout(self.pos_encoding(tgt) + tgt)
        x = self.emb2hid(combined)
        x = x.permute(0, 2, 1)
        batch_size, hidden_dim, ft_size = x.shape
        attn = torch.zeros((batch_size, 1), device=tgt.device)
        self.scale = self.scale.to(tgt.device)

        for i, conv in enumerate(self.conv):
            x = self.dropout(x)
            pad = torch.zeros(batch_size, hidden_dim, self.kernel_size - 1).fill_(self.tgt_pad_idx).to(x.device)
            x = torch.cat((pad, x), dim=2)
            x = conv(x)
            x = F.glu(x, dim=1)
            attn, convd = self._calc_attn(combined, x, enc_conv, enc_comb)
            x = (x + convd) * self.scale
        x = self.hid2emb(x.permute(0, 2, 1))
        return x, attn

    def _calc_attn(self, x, convd, enc_conv, enc_comb):
        conv_x = self.attn_hid2emb(convd.permute(0, 2, 1))
        comb = (conv_x + x) * self.scale

        energy = torch.matmul(comb, enc_conv.permute(0, 2, 1))
        attn = F.softmax(energy, dim=2)
        attn_encoding = torch.matmul(attn, enc_comb)
        attn_encoding = self.attn_emb2hid(attn_encoding)
        attn_comb = (convd + attn_encoding.permute(0, 2, 1)) * self.scale
        return attn, attn_comb


class ConvSeq2Seq(nn.Module):
    def __init__(self,
                 feature_size,
                 hidden_dim,
                 num_layers,
                 kernel_size,
                 tgt_pad_idx,
                 dropout,
                 max_len):
        super().__init__()
        self.encoder = Encoder(feature_size=feature_size,
                               hidden_dim=hidden_dim,
                               num_layers=num_layers,
                               kernel_size=kernel_size,
                               dropout=dropout,
                               max_len=max_len)
        self.decoder = Decoder(feature_size=feature_size,
                               hidden_dim=hidden_dim,
                               num_layers=num_layers,
                               kernel_size=kernel_size,
                               tgt_pad_idx=tgt_pad_idx,
                               dropout=dropout,
                               max_len=max_len)

    def forward(self, x: torch.Tensor, trg: torch.Tensor):
        enc_conv, enc_comb = self.encoder(x)
        out, attn = self.decoder(trg, enc_conv, enc_comb)
        return out, attn
