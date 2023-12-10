import os
import glob
import numpy as np
import librosa
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision.transforms import Compose
from rmvpe import rmvpe
from alaapNet.tools.utils import Util
from alaapNet.data.transforms import ToTensor, FillGaps

'''
audio file naming convention: <raga_id>-<midi_key>-<artist_id>.wav
'''


class AlapanaDataset(Dataset):
    def __init__(self,
                 root: str,
                 train: bool,
                 input_length,
                 target_length,
                 hop_length,
                 resample_factor,
                 max_value,
                 device='cpu'):
        super().__init__()
        self.src_path = os.path.join(root, "audio", "train" if train else "test")
        self.pitch_path = os.path.join(root, "pitch", "train" if train else "test")
        self.input_length = input_length
        self.target_length = target_length
        self.hop_length = hop_length
        self.fs = 100 * resample_factor
        self.max_value = max_value
        rmvpe_model_path = os.path.join("rmvpe", "rmvpe.pt")
        self.rmvpe = rmvpe.RMVPE(rmvpe_model_path, is_half=False, device=device)
        self.transforms = Compose([FillGaps(), ToTensor()])

        self.recordings = []
        self.data = self.__preprocess_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.transforms(self.data[idx][0]).contiguous().float()
        y = self.transforms(self.data[idx][1]).contiguous().float()
        return X, y

    def __preprocess_data(self):
        for d in os.listdir(self.pitch_path):
            if os.path.isdir(os.path.join(self.pitch_path, d)):
                self.recordings.append(d)

        data = []
        for directory in self.recordings:
            paths = glob.glob(os.path.join(self.pitch_path, directory, "*.txt"))
            for i in range(len(paths)):
                phrase = np.loadtxt(os.path.join(self.pitch_path, directory, f"{i}.txt"))
                phrase = phrase / self.max_value
                phrase[np.abs(phrase) == np.inf] = 0
                phrase = librosa.resample(phrase, orig_sr=100, target_sr=self.fs).astype(np.float32)
                data.extend(self.__split_chunks(phrase))
        return data

    def __split_chunks(self, phrase: np.ndarray):
        """
        Split a phrase into input and target lengths
        :param phrase:
        :return: list of (X, y)
        """
        num_points = (len(phrase) - self.input_length - self.target_length + self.hop_length)
        padded = np.hstack([phrase, np.zeros(self.input_length + self.hop_length + self.target_length)])
        padded = padded.reshape(-1, 1)
        chunks = []
        for i in range(0, num_points, self.hop_length):
            x = padded[i: i + self.input_length]
            y = padded[i + self.input_length: i + self.input_length + self.target_length]
            chunks.append((x, y))
        return chunks

    def __extract_and_save_pitches(self):
        # Create pitch directory to store pitches
        if not os.path.exists(self.pitch_path):
            os.makedirs(self.pitch_path)
        for file in glob.glob(os.path.join(self.src_path, "*.wav")):
            filename, (raga, key, artist) = Util.unpack_filename(file)
            audio, fs = librosa.load(file, sr=16000, mono=True)
            print(file)
            chunks = self.__split_audio(audio, fs)
            path = os.path.join(self.pitch_path, filename)

            if os.path.exists(path):
                continue

            os.mkdir(path)
            for i, chunk in tqdm(enumerate(chunks), total=len(chunks)):
                pitch = self.__extract_pitch(chunk, midi_key=key)
                np.savetxt(os.path.join(path, f"{i}.txt"), pitch)

    @staticmethod
    def __split_audio(audio, fs, num_zeros=5, min_samples=44100, hop_sec=0.025, block_sec=0.05, eps=1e-6):
        hop = int(hop_sec * fs)
        e = librosa.feature.rms(y=audio, frame_length=int(block_sec * fs), hop_length=hop)[0]
        b = []
        i = 0
        while i < len(e) - num_zeros:
            if np.mean(e[i:i + num_zeros]) < eps:
                temp = i
                while np.mean(e[i:i + num_zeros]) < eps:
                    i += 1
                b.append((int(temp * hop), int(i * hop)))
            else:
                i += 1

        b = np.array(b)
        splits = []
        t = audio[:b[0, 0]]
        if len(t) > min_samples:
            splits.append(t)
        for i in range(len(b) - 1):
            t = audio[b[i, 1]: b[i + 1, 0]]
            if len(t) > min_samples:
                splits.append(t)

        t = audio[b[-1, 1]:]
        if len(t) > min_samples:
            splits.append(t)

        return splits

    def __extract_pitch(self, audio: np.ndarray, midi_key, threshold=0.3):
        f0 = self.rmvpe.infer_from_audio(audio, thred=threshold)
        midi = librosa.hz_to_midi(f0)
        midi = Util.normalize_midi(midi, midi_key=midi_key)
        midi = self.__truncate_pitches(midi)
        return midi

    @staticmethod
    def __truncate_pitches(midi):
        # Truncate start
        start_idx = 0
        for i in range(len(midi)):
            if not (midi[i] < 1e-3 or np.isnan(midi[i]) or np.isinf(midi[i])):
                start_idx = i
                break

        # Truncate end
        end_idx = len(midi) - 1
        for i in range(len(midi) - 1, -1, -1):
            if not (midi[i] < 1e-3 or np.isnan(midi[i]) or np.isinf(midi[i])):
                end_idx = i
                break

        return midi[start_idx: end_idx]


class STADataset(Dataset):
    def __init__(self):
        super().__init__()