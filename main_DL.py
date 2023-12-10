import matplotlib.pyplot as plt
import librosa
import numpy as np
import scipy.signal

from alaapNet.data.dataset import AlapanaDataset    #, AlapanaDataLoader, AlapanaDataLoader_bits
from torch.utils.data import DataLoader
from alaapNet.tools.train import AlapanaTrainer
# from alaapNet.eval.eval import InferenceEngine
from alaapNet.tools.utils import Util
# from alaapNet.models.conv_seq2seq import ConvSeq2Seq
from alaapNet.models.alaapNet import ConvNet
from tqdm import tqdm
import torch
import sys
import os

torch.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":
    # dataset = AlapanaDataset("dataset", train=True, resample_factor=1, max_value=3.0)
    # loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # for X, y in loader:
    #     print(X.shape, y.shape)
    #     plt.plot(X.flatten())
    #     plt.plot(y.flatten())
    #     plt.show()

    in_seq_len = 100
    out_seq_len = 100
    hop_length = 5
    hidden_size = 200
    num_layers = 4
    batch_size = 128
    lr = 1e-3
    trainer = AlapanaTrainer(dataset_path="dataset",
                             in_seq_len=in_seq_len,
                             out_seq_len=out_seq_len,
                             hop_length=hop_length,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_size=batch_size,
                             lr=lr,
                             resample_factor=1,
                             max_value=3.0)

    trainer.train(epochs=10000, ckpt_dir="checkpoints")
    # trainer.eval(ckpt_path="checkpoints/checkpoint_conv.pth")

    # batch_size = 10
    # seq_len_x = 100
    # seq_len_y = 50
    # X = torch.randn(size=(batch_size, seq_len_x, 1))
    # y = torch.randn(size=(batch_size, seq_len_y, 1))
    # model = ConvNet(out_seq_len=seq_len_y)
    # out = model(X)
    # model = ConvSeq2Seq(bits, feature_size, num_layers, k_size, 0, 0, 5000)
    #
    # out = model(X, y)

    # x = torch.tensor([[0.01, 0.5, 0.49]])
    # y = torch.tensor([[1/3, 1/3, 1/3]])
    # print(Util.js_div(x, target=y))
