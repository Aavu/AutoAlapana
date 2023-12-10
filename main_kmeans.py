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
    bits = 6
    train_data = HistogramDataset(root="dataset", train=True)
    model = KMeans(k=50, distance_fn=JSDivergence(), max_iter=1000)
    train_x = train_data.get_histograms(max_val=0.25, bits=bits)
    model.fit(train_x, checkpoint_path="checkpoints/KMeans_diff")
