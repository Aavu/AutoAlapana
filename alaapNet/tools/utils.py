import pickle
import librosa
import numpy as np
import os
import json

import torch


class Util:
    @staticmethod
    def normalize_midi(pitches, midi_key: int):
        out = 1 + (pitches - midi_key) / 12.0
        # out[out < 0] = 0
        # out[np.isnan(pitches)] = 0
        return out

    @staticmethod
    def unpack_filename(filepath: str) -> tuple:
        """
        parse filepath for raga, key and artist
        :param filepath: file path
        :return: filename, raga-id, key, artist-id
        """
        filename = os.path.split(filepath)[-1]  # get just the filename
        filename = os.path.splitext(filename)[0]  # remove .wav/.midi
        return filename, tuple(map(int, filename.split('-')))

    @staticmethod
    def pkl_load(filename: str):
        with open(filename, 'rb') as handle:
            return pickle.load(handle)

    @staticmethod
    def pkl_dump(filename: str, data):
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def to_numpy(x: torch.Tensor):
        return x.cpu().detach().numpy()

    @staticmethod
    def zero_lpf(x: torch.Tensor, alpha, device='cpu'):
        x0 = torch.clone(x)
        for i in range(1, len(x)):
            x0[i] = (alpha * x0[i - 1]) + ((1 - alpha) * x0[i])

        x0 = torch.flip(x0, dims=(0,))
        for i in range(1, len(x)):
            x0[i] = (alpha * x0[i - 1]) + ((1 - alpha) * x0[i])
        x0 = torch.flip(x0, dims=(0,))

        # restore NULL values
        x0[x < 0.01] = 0
        return x0
