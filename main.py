import librosa
import numpy as np
import torch

from alapana_nn.utils import Util
from matplotlib import pyplot as plt
from alapana_nn.dataset import HistogramDataset, SilenceDataset, SilenceDataloader
from alapana_nn.train import SilenceTrainer
from alapana_nn.ml import KMeans, JSDivergence, SqL2
import sys
import alapana_nn.synth as synth
import soundfile as sf

np.set_printoptions(precision=4, threshold=sys.maxsize)

if __name__ == "__main__":
    # data = SilenceDataset(root="dataset", train=True, divisions=1)

    # batch_size = 32
    # lr = 0.001
    # hidden_size = 64
    # num_layers = 4
    # trainer = SilenceTrainer(dataset_path="dataset",
    #                          batch_size=batch_size,
    #                          lr=lr,
    #                          hidden_size=hidden_size,
    #                          num_layers=num_layers)

    # trainer.train(epochs=20000)
    # trainer.eval(ckpt_path="checkpoints/checkpoint-silence.pth")

    bits = 6
    train_data = HistogramDataset(root="dataset", train=True, device='cuda')
    # for s, x in train_data:
    #     print(s)
    #     plt.plot(x)
    #     plt.show()
    # max_val = .25
    # for x, h, p in train_data:
    #     diff = np.diff(x)
    #     diff = (max_val + diff) / (2*max_val)
    #     temp = np.clip(diff, a_min=0, a_max=1)
    #     hist = Util.compute_histogram(temp, bits, max_val=1)
    #     plt.subplot(411)
    #     plt.plot(x)
    #     plt.subplot(412)
    #     plt.plot(diff)
    #     plt.subplot(413)
    #     plt.plot(h)
    #     plt.subplot(414)
    #     plt.plot(hist)
    #     plt.show()

    loss_fn = JSDivergence()
    model = KMeans(k=100, distance_fn=loss_fn, max_iter=100)
    model.fit(train_data.get_histograms(bits), checkpoint_path="checkpoints/KMeans_diff1")
