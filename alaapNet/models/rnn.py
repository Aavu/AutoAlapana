import torch
from torch import nn
from torch.nn import functional as F
import random
from alaapNet.tools.utils import Util


class GRUTimeSeries(nn.Module):
    def __init__(self, hidden_size, num_recurrent_layers, num_fc_layers, out_seq_len):
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, num_recurrent_layers, batch_first=True)
        fc = []
        for i in range(num_fc_layers-1):
            fc.append(nn.Linear(hidden_size, hidden_size))
            fc.append(nn.ReLU())

        fc.extend([nn.Linear(hidden_size, out_seq_len), nn.Sigmoid()])
        self.fc = nn.Sequential(*fc)

    def forward(self, x: torch.Tensor):
        x, hidden = self.gru(x)
        x = self.fc(x[:, -1])
        return x.unsqueeze(-1)
