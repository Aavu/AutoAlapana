import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size=feature_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        nn.init.xavier_uniform_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        nn.init.constant_(self.gru.bias_ih_l0, 0)
        nn.init.constant_(self.gru.bias_hh_l0, 0)

    def forward(self, x: torch.Tensor):
        h0 = torch.zeros((self.gru.num_layers, len(x), self.gru.hidden_size), requires_grad=True, device=x.device)
        out, hidden = self.gru(x, h0.detach())
        return out, hidden


class SilencePredictorModel(nn.Module):
    def __init__(self, feature_size, output_size, hidden_size, num_layers, dropout_p=0.25):
        super().__init__()
        self.encoder = Encoder(feature_size, hidden_size, num_layers)
        self.fc = []

        # self.fc.append(nn.Dropout(dropout_p))
        self.fc.append(nn.Linear(2 * (hidden_size * num_layers), hidden_size))
        self.fc.append(nn.ReLU())

        for i in range(num_layers - 1):
            # self.fc.append(nn.Dropout(dropout_p))
            self.fc.append(nn.Linear(hidden_size, hidden_size))
            self.fc.append(nn.ReLU())

        # self.fc.append(nn.Dropout(dropout_p))
        self.fc.append(nn.Linear(hidden_size, output_size))

        self.fc = nn.Sequential(*self.fc)

    def forward(self, current_phrase: torch.Tensor, next_phrase: torch.Tensor):
        current_phrase = current_phrase[:, ::2]
        next_phrase = next_phrase[:, ::2]
        _, hidden_current = self.encoder(current_phrase)
        _, hidden_next = self.encoder(next_phrase)
        batch_size = len(current_phrase)
        hidden_current = hidden_current.permute(1, 0, 2).view(batch_size, -1)
        hidden_next = hidden_next.permute(1, 0, 2).view(batch_size, -1)
        hidden = torch.cat([hidden_current, hidden_next], dim=-1)
        x = self.fc(hidden)
        return x


class SilencePredictorModel2(nn.Module):
    def __init__(self, feature_size, output_size, hidden_size, num_layers, dropout_p=0.25):
        super().__init__()
        self.encoder = Encoder(128, hidden_size, num_layers)

        self.conv = []

        self.conv.append(nn.Dropout(dropout_p))
        self.conv.append(nn.Conv1d(feature_size, 16, 9))
        self.conv.append(nn.AvgPool1d(3))
        self.conv.append(nn.ReLU())
        # self.conv.append(nn.Dropout(dropout_p))
        self.conv.append(nn.Conv1d(16, 32, 7))
        self.conv.append(nn.AvgPool1d(3))
        self.conv.append(nn.ReLU())
        # self.conv.append(nn.Dropout(dropout_p))
        self.conv.append(nn.Conv1d(32, 64, 5))
        self.conv.append(nn.AvgPool1d(3))
        self.conv.append(nn.ReLU())
        # self.conv.append(nn.Dropout(dropout_p))
        self.conv.append(nn.Conv1d(64, 128, 3))
        self.conv.append(nn.AvgPool1d(3))
        self.conv.append(nn.ReLU())
        self.conv = nn.Sequential(*self.conv)

        self.fc = []

        # self.fc.append(nn.Dropout(dropout_p))
        self.fc.append(nn.Linear((2*hidden_size) + 2, hidden_size))
        self.fc.append(nn.ReLU())

        self.fc.append(nn.Dropout(dropout_p))
        self.fc.append(nn.Linear(hidden_size, output_size))
        self.fc.append(nn.ReLU())
        self.fc = nn.Sequential(*self.fc)

    def forward(self, current_phrase: torch.Tensor, next_phrase: torch.Tensor, lengths: torch.Tensor):
        current_phrase = current_phrase.permute(0, 2, 1)
        next_phrase = next_phrase.permute(0, 2, 1)
        current_phrase = self.conv(current_phrase)
        next_phrase = self.conv(next_phrase)
        current_phrase = current_phrase.permute(0, 2, 1)
        next_phrase = next_phrase.permute(0, 2, 1)

        out_current, hidden_current = self.encoder(current_phrase)
        out_next, hidden_next = self.encoder(next_phrase)
        out_current = out_current[:, -1]
        out_next = out_next[:, -1]

        # batch_size = len(current_phrase)
        # hidden_current = hidden_current.permute(1, 0, 2).contiguous().view(batch_size, -1)
        # hidden_next = hidden_next.permute(1, 0, 2).contiguous().view(batch_size, -1)
        out = torch.cat([out_current, out_next, lengths], dim=-1)
        x = self.fc(out)
        return x
