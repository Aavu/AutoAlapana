import matplotlib.pyplot as plt

from alaapNet.models.alaapNet import Seq2Seq, Seq2SeqComponent, TransformerModel
from alaapNet.data.dataset import AlapanaDataset, AlapanaDataLoader, AlapanaDataLoader_bits
from alaapNet.tools.utils import Util
import numpy as np
import torch
import os


class InferenceEngine:
    def __init__(self, ckpt_path: str, input_size, feature_size, num_layers, n_heads):
        self.ckpt_path = ckpt_path

        self.input_size = input_size
        self.feature_size = feature_size
        self.num_layers = num_layers
        self.n_heads = n_heads

        self.batch_size = 1

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available else 'cpu')

        # self.model = Seq2SeqComponent(self.feature_size,
        #                               hidden_size=self.hidden_size,
        #                               num_layers=self.num_layers,
        #                               bidirectional_encoder=self.bidirectional).to(self.device)

        self.model = TransformerModel(feature_size=self.input_size,
                                      d_model=self.feature_size,
                                      n_heads=self.n_heads,
                                      num_layers=self.num_layers).to(self.device)
        self.ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(self.ckpt['model_state'])
        self.model.eval()

    def eval(self, dataset_path, train_data=False):
        data = AlapanaDataset(dataset_path, train=train_data)
        loader = AlapanaDataLoader(data, feature_size=self.input_size, batch_size=self.batch_size, device=self.device, shuffle=False)
        # loader = AlapanaDataLoader_bits(data, bits=8, batch_size=1, max_val=3, device=self.device, shuffle=False)
        i = 0
        for X, y in loader:
            X = X.to(self.device)
            y = y.to(self.device)
            y_input = y[:, :-1]
            y_expected = y[:, 1:]
            tgt_mask = self.model.get_tgt_mask(y_input.size(1)).to(self.device)
            # x_pad_mask = self.model.create_pad_mask(X, pad_token=-1)
            # tgt_pad_mask = self.model.create_pad_mask(y_input, pad_token=-1)
            pred = self.model(X, tgt=y_input,
                              tgt_mask=None)
            plt.plot(Util.to_numpy(y_expected[0].view(-1)))
            plt.plot(Util.to_numpy(pred[0].view(-1)))
            plt.show()
            if i == 50:
                break
            i += 1
