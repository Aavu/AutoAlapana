import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class BaseModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.in_size = input_size
        self.config = {}
        self.last_size = 0

    @staticmethod
    def get_out_size(in_size, kernel_size, padding=0, stride=1, dilation=1):
        return int(np.floor((in_size + 2*padding - dilation*(kernel_size - 1) - 1) / stride) + 1)

    def load_checkpoint(self, path, move_to_device=True):
        # Set Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(path, map_location=device)['model_state'])
        if move_to_device:
            self.to(device)

    @staticmethod
    def get_activation(activation: str, param: float = None):
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        if activation == "tanh":
            return nn.Tanh()
        if activation == "leakyrelu":
            if param is not None:
                return nn.LeakyReLU(param)
            return nn.LeakyReLU()

    def compute_last_size(self, kernel_size):
        size = self.get_out_size(self.in_size[-1], kernel_size[0], stride=self.stride)
        for i in range(1, len(kernel_size)):
            size = self.get_out_size(size, kernel_size[i])
        self.last_size = size


class GamakaNet(BaseModel):
    def __init__(self, input_size,
                 kernel_size=(5, 9, 11, 13, 15, 17, 19, 21),
                 ch_sizes: tuple = (1, 10, 10, 20, 20, 30, 30, 40, 40),
                 latent_size: int = 150, activation: str = "relu"):

        super().__init__(input_size)
        self.ch_size = ch_sizes
        self.latent_size = latent_size

        self.stride = 3
        self.conv = nn.ModuleList(
            [nn.Conv1d(ch_sizes[0], ch_sizes[1], (kernel_size[0],), stride=(self.stride,))]
        )
        for i in range(1, len(ch_sizes) - 1):
            self.conv.append(nn.Conv1d(ch_sizes[i], ch_sizes[i + 1], (kernel_size[i],)))

        self.last_size = 81
        # self.compute_last_size(kernel_size)

        self.e_fc = nn.Linear(ch_sizes[-1] * self.last_size, latent_size)
        self.d_fc = nn.Linear(latent_size, ch_sizes[-1] * self.last_size)

        self.tConv = nn.ModuleList(
            [nn.ConvTranspose1d(ch_sizes[1], ch_sizes[0], (kernel_size[0],), stride=(self.stride,), dilation=2, padding=1)]
        )

        for i in range(1, len(ch_sizes) - 1):
            self.tConv.append(nn.ConvTranspose1d(ch_sizes[i + 1], ch_sizes[i], (kernel_size[i],)))

        self.ea = F.sigmoid # self.get_activation(activation)
        self.da = self.ea

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, ft_size = x.shape
        x = x.permute(0, 2, 1)
        x = self.ea(self.conv[0](x))
        x = self.ea(self.conv[1](x))
        x_skip1 = x.clone()
        x = self.ea(self.conv[2](x))
        x = self.ea(self.conv[3](x))
        x_skip3 = x.clone()
        x = self.ea(self.conv[4](x))
        x = self.ea(self.conv[5](x))
        x_skip5 = x.clone()
        x = self.ea(self.conv[6](x))
        x = self.ea(self.conv[7](x))
        x_skip6 = x.clone()
        x = x.view(batch_size, self.ch_size[-1] * self.last_size)

        z = self.ea(self.e_fc(x))
        x = self.da(self.d_fc(z))

        x = x.view(batch_size, self.ch_size[-1], self.last_size)
        x = x + x_skip6
        x = self.da(self.tConv[7](x))
        x = self.da(self.tConv[6](x))
        x = x + x_skip5
        x = self.da(self.tConv[5](x))
        x = self.da(self.tConv[4](x))
        x = x + x_skip3
        x = self.da(self.tConv[3](x))
        x = self.da(self.tConv[2](x))
        x = x + x_skip1
        x = self.da(self.tConv[1](x))
        x = self.da(self.tConv[0](x))
        x = x.permute(0, 2, 1)
        return x
