import matplotlib.pyplot as plt
import librosa
import numpy as np
import scipy.signal

from alaapNet.data.dataset import AlapanaDataset    #, AlapanaDataLoader, AlapanaDataLoader_bits
from torch.utils.data import DataLoader
from alaapNet.tools.train import AlapanaTrainer
# from alaapNet.eval.eval import InferenceEngine
# from alaapNet.tools.utils import Util
# from alaapNet.models.conv_seq2seq import ConvSeq2Seq
from tqdm import tqdm
import torch
import sys
import os

torch.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":
    # dataset = AlapanaDataset("dataset", resample_factor=1, max_value=3.0)
    # loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # for X, y in loader:
    #     print(X.shape, y.shape)
    #     plt.plot(X.flatten())
    #     plt.plot(y.flatten())
    #     plt.show()

    in_seq_len = 50
    out_seq_len = 25
    hop_length = 10
    hidden_size = 200
    num_layers = 4
    batch_size = 128
    lr = 5e-4
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

    # trainer.train(epochs=10000, ckpt_dir="checkpoints")
    trainer.eval(ckpt_path="checkpoints/checkpoint.pth")

    # batch_size = 10
    # seq_len_x = 128
    # seq_len_y = 256
    # X = torch.randn(size=(batch_size, seq_len_x, bits))
    # y = torch.randn(size=(batch_size, seq_len_y, bits))
    # model = ConvSeq2Seq(bits, feature_size, num_layers, k_size, 0, 0, 5000)
    #
    # out = model(X, y)
