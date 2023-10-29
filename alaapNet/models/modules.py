import torch
from torch import nn
from alaapNet.tools.utils import Util
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)


class AlaapLoss(nn.Module):
    def __init__(self, alpha=0.9):
        super().__init__()
        self.mse_t = nn.MSELoss()
        self.mse_f = nn.MSELoss()
        self.alpha = alpha

    # def forward(self, pred: torch.Tensor, target: torch.Tensor):
    #     dev = pred.device
    #     if dev == torch.device('mps:0'):
    #         pred = pred.cpu()
    #         target = target.cpu()
    #
    #     pred_fft = torch.abs(torch.fft.rfft(pred)).to(dev)
    #     target_fft = torch.abs(torch.fft.rfft(target)).to(dev)
    #     return self.mse_f(pred_fft, target_fft) + self.mse_t(pred, target)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        dev = pred.device
        if dev == torch.device('mps:0'):
            pred = pred.cpu()
            target = target.cpu()

        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        trend = Util.zero_lpf(pred, self.alpha)
        seasonal = pred - trend
        pred_stft = torch.abs(torch.stft(seasonal, 256, 4, 64, return_complex=True)).to(dev)

        trend = Util.zero_lpf(target, self.alpha)
        seasonal = pred - trend
        target_stft = torch.abs(torch.stft(seasonal, 256, 4, 64, return_complex=True)).to(dev)

        return self.mse_f(pred_stft, target_stft) + self.mse_t(pred, target)
