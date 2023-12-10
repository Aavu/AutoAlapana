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
    def unnormalize_midi(values, midi_key: int):
        return midi_key + ((values - 1) * 12)

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
    def zero_lpf(x: torch.Tensor, alpha, restore_zeros=True, device='cpu'):
        x0 = torch.clone(x)
        for i in range(1, len(x)):
            x0[i] = (alpha * x0[i - 1]) + ((1 - alpha) * x0[i])

        x0 = torch.flip(x0, dims=(0,))
        for i in range(1, len(x)):
            x0[i] = (alpha * x0[i - 1]) + ((1 - alpha) * x0[i])
        x0 = torch.flip(x0, dims=(0,))

        # restore NULL values
        if restore_zeros:
            x0[x < 0.01] = 0
        return x0

    @staticmethod
    def js_div(x: np.ndarray, target: np.ndarray, eps=1e-6) -> float:
        """
        Jensen-Shannon Divergence
        :param x: observed distribution
        :param target: actual distribution
        :param eps: Small value to make sure result is not nan
        :return: scalar. 0 -> similar distribution, +ve means distribution diverge
        """
        x = torch.from_numpy(x) + eps
        target = torch.from_numpy(target) + eps
        p = 0.5 * (x + target)
        return 0.5 * (torch.nn.functional.kl_div(x.log(), p) + torch.nn.functional.kl_div(target.log(), p)) / np.log(2)

    @staticmethod
    def bit_crush(x: np.ndarray, bits, max_val):
        out = x[:]
        out = out / max_val
        assert np.min(out) >= 0 and np.max(out) < 1, f"min = {np.min(out)}, max = {np.max(out)}"
        out *= 2 ** bits
        out = np.round(out).astype(int)
        return out

    @staticmethod
    def histogram(x: np.ndarray, bins):
        out = np.zeros(bins, dtype=float)

        for i in x:
            if i >= bins:
                raise Exception(f"Values exceed bin count, i: {i}, bins: {bins}")
            out[i] += 1

        out = out / len(x)
        return out

    @staticmethod
    def compute_histogram(x, bits, max_val=3.0):
        x_bit = Util.bit_crush(x, bits=bits, max_val=max_val)
        hist = Util.histogram(x_bit, bins=2 ** bits)
        return hist
