import os
import sys
import glob
import numpy as np
import librosa
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from torch import nn as nn
from torch.nn import functional as F
from torchvision.transforms import Compose
from alapana_nn.utils import Util
from alaapNet.data.transforms import ToTensor, FillGaps
import soundfile as sf
from alapana_nn.pitchTrack import PitchTrack


class BaseDataset(Dataset):
    def __init__(self, root: str,
                 train: bool, pitch_detection_algorithm: str = 'crepe', device='cpu'):
        self.root_path = root
        self.train = train
        self.audio_path = os.path.join(root, "audio", "train" if train else "test")
        self.pitch_path = os.path.join(root, "pitch", "train" if train else "test")
        self.recordings = self.__populate_recordings()
        self.data = []

        self.pitch_detection_algorithm = pitch_detection_algorithm
        self.pitch_tracker = PitchTrack(pitch_detection_algorithm, fmin=librosa.note_to_hz('E1'),
                                        fmax=librosa.note_to_hz('A6'), device=device)

        self.max_silence_length = 0
        self.is_discrete = False
        self.divisions = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __populate_recordings(self):
        rec = []
        for d in os.listdir(self.audio_path):
            if os.path.splitext(os.path.join(self.audio_path, d))[1] == '.wav':
                name = os.path.splitext(d)[0]
                rec.append(name)
        return rec

    def extract_pitch(self, chunks: List[Tuple[np.ndarray, int]], fs, key) -> List[Tuple[np.ndarray, int]]:
        out = []
        for c, s in tqdm(chunks, total=len(chunks)):
            p = Util.extract_pitch(c, fs, key, pitch_tracker=self.pitch_tracker)
            out.append((p, s))

        return out

    @staticmethod
    def get_max_lengths(chunks: List[Tuple[np.ndarray, np.ndarray, int]]) -> Tuple[int, int]:
        max_silence = 0
        max_phrase = 0
        for x1, x2, silence_length in chunks:
            max_silence = max(max_silence, silence_length)
            max_phrase = max(max_phrase, len(x1), len(x2))
        return max_phrase, max_silence

    def quantize_length(self, length):
        assert self.max_silence_length > 0, f"Max silence length = {self.max_silence_length}"
        if self.is_discrete:
            return np.array([length / self.max_silence_length]).reshape(1, -1)
        #    l = x * max_l / div
        # => x = l * div / max_l
        div = self.divisions - 1  # to include zeros
        x = min(div, int(np.round((length / self.max_silence_length) * div)))
        return x


class HistogramDataset(BaseDataset):
    def __init__(self, root: str,
                 train: bool,
                 divisions: int or None = None,
                 pitch_detection_algorithm='crepe',
                 device='cpu'):
        super().__init__(root, train, pitch_detection_algorithm=pitch_detection_algorithm, device=device)

        self.pkl_path = os.path.join(self.root_path, f"silence_{'train' if self.train else 'test'}_crepe.pkl")
        if not os.path.exists(self.pkl_path):
            self.__preprocess_data()

        self.data = Util.pkl_load(self.pkl_path)
        self.max_phrase_length, self.max_silence_length = self.get_max_lengths(self.data)
        self.divisions = divisions
        self.is_discrete = self.divisions is None or self.divisions == 0 or self.divisions == 1

    def get_histograms(self, bits, max_val=3.0, derivative_max_val=0.25):
        data = np.zeros((len(self), 2, 2 ** bits))
        for i, (s0, x, s1) in enumerate(self.data):
            data[i] = Util.compute_2d_histogram(x, bits, max_val, derivative_max_val, fill_gaps=True)
        return data

    def __getitem__(self, idx):
        s0, x, s1 = self.data[idx]
        return self.quantize_length(s0), x, self.quantize_length(s1)

    @staticmethod
    def get_max_lengths(chunks: List[Tuple[int, np.ndarray]]) -> Tuple[int, int]:
        max_silence = 0
        max_phrase = 0
        for (s0, x, s1) in chunks:
            max_silence = max(max_silence, s0, s1)
            max_phrase = max(max_phrase, len(x))
        return max_phrase, max_silence

    def __preprocess_data(self):
        data = []
        for file in self.recordings:
            filename, (raga, key, artist) = Util.unpack_filename(file)
            print(f"{filename}")
            path = os.path.join(self.audio_path, f"{file}.wav")
            print(f" - loading {filename}")
            audio, fs = librosa.load(path, sr=44100)
            print(f" - Getting silence bounds")
            silence_bounds = Util.get_silence_bounds(audio, fs)
            print(f" - Splitting into chunks")
            audio_chunks = Util.split(audio, silence_bounds, min_samples=44100)
            print(f" - Extracting pitch")
            chunks = self.extract_pitch(audio_chunks, fs, key)

            data.append((0, chunks[0][0], chunks[0][1]))
            for i in range(1, len(chunks)):
                c1 = chunks[i - 1]
                c2 = chunks[i]
                # progress = i / len(chunks)
                data.append((c1[1], c2[0], c2[1]))
        self.data = data
        Util.pkl_dump(self.pkl_path, self.data)


class SilenceDataset(BaseDataset):
    def __init__(self, root: str,
                 train: bool,
                 divisions: int or None = None):
        """
        Dataset to predict length of silence between 2 phrases
        :param root: path to dataset root
        :param train: Whether it is train set or test set
        :param divisions: number of divisions for silence. Minimum silence length would be (1/div * max silence)
        """
        super().__init__(root, train)
        self.divisions = divisions
        self.is_discrete = self.divisions is None or self.divisions == 0 or self.divisions == 1

        self.pkl_path = os.path.join(self.root_path, f"{'train' if self.train else 'test'}.pkl")
        if not os.path.exists(self.pkl_path):
            self.__preprocess_data()

        self.data = Util.pkl_load(self.pkl_path)
        self.max_phrase_length, self.max_silence_length = self.get_max_lengths(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c1, c2, s = self.data[idx]
        c1 = torch.from_numpy(c1).reshape(-1, 1)
        c2 = torch.from_numpy(c2).reshape(-1, 1)
        lengths = torch.from_numpy(np.array([len(c2), len(c2)]))
        return c1, c2, torch.from_numpy(self.quantize_length(s)), lengths

    def __preprocess_data(self):
        data = []
        for file in self.recordings:
            filename, (raga, key, artist) = Util.unpack_filename(file)
            print(f"{filename}")
            path = os.path.join(self.audio_path, f"{file}.wav")
            print(f" - loading {filename}")
            audio, fs = librosa.load(path, sr=44100)
            print(f" - Getting silence bounds")
            silence_bounds = Util.get_silence_bounds(audio, fs)
            print(f" - Splitting into chunks")
            audio_chunks = Util.split(audio, silence_bounds, min_samples=44100)
            print(f" - Extracting pitch")
            chunks = self.extract_pitch(audio_chunks, fs, key)

            for i in range(len(chunks) - 1):
                c1 = chunks[i]
                c2 = chunks[i + 1]
                data.append((c1[0], c2[0], c1[1]))
        self.data = data
        Util.pkl_dump(self.pkl_path, self.data)


class SilenceDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, collate_fn=self.collate)

    @staticmethod
    def collate(batch):
        temp1 = []
        temp2 = []
        y_batch = []
        lengths = []
        for x1, x2, y, l in batch:
            temp1.append(x1)
            temp2.append(x2)
            y_batch.append(y)
            lengths.append(l)
        x1 = nn.utils.rnn.pad_sequence(temp1, batch_first=True)
        x2 = nn.utils.rnn.pad_sequence(temp2, batch_first=True)
        y_batch = torch.cat(y_batch, dim=0).float()
        lengths = torch.stack(lengths)
        return x1, x2, y_batch, lengths
