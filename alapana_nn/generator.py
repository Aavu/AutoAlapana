import librosa
import numpy as np
from alapana_nn.utils import Util
from matplotlib import pyplot as plt
from alapana_nn.dataset import HistogramDataset, SilenceDataset, SilenceDataloader
from alapana_nn.train import SilenceTrainer
from alapana_nn.ml import KMeans, JSDivergence, SqL2
import sys
import soundfile as sf
from typing import List, Tuple
from alapana_nn.pitchTrack import PitchTrack
from alapana_nn.models import SilencePredictorModel
import torch
from tqdm import tqdm
import time


class AlapanaGenerator:
    def __init__(self,
                 library_path="dataset",
                 kmeans_ckpt_path="checkpoints/KMeans_diff1",
                 silence_ckpt_path="checkpoints/checkpoint-silence.pth",
                 target_midi_key: int = 62,
                 pitch_detection_algorithm: str = 'crepe', device='cpu'):
        self.loss_fn = JSDivergence()
        self.device = device
        self.kmeans_model = KMeans(k=50, distance_fn=self.loss_fn, ckpt_path=kmeans_ckpt_path)
        self.bits = 6
        self.divisions = 1
        self.library = HistogramDataset(root=library_path,
                                        train=True,
                                        divisions=self.divisions,
                                        device=device)
        self.max_silence_in_samples = self.library.max_silence_length
        self.target_midi_key = target_midi_key
        self.pitch_tracker = PitchTrack(pitch_detection_algorithm, fmin=librosa.note_to_hz('E1'),
                                        fmax=librosa.note_to_hz('A6'), device=device)

        # helpers
        self.used_idx = []

        # Silence prediction stuffs
        # feature_size = 1
        # hidden_size = 128
        # num_layers = 4
        # self.silence_model = SilencePredictorModel(feature_size=feature_size,
        #                                            output_size=self.divisions,
        #                                            hidden_size=hidden_size,
        #                                            num_layers=num_layers).to(device)
        # ckpt = torch.load(silence_ckpt_path, map_location=self.device)
        # self.silence_model.load_state_dict(ckpt['model_state'])
        # self.max_silence_in_samples = ckpt['max_silence']

    def generate_alapana_for(self, audio: np.ndarray, fs: float, midi_key: int) -> List[Tuple[int, np.ndarray]]:
        print("Preprocessing audio")
        query_phrases = self.__preprocess_audio(audio, fs, midi_key)
        print("Generating phrases")
        start = time.time()
        phrases = self.generate_phrases(query_phrases)
        print(f"Time elapsed: {time.time() - start} sec")
        return phrases  # self.__add_silences(phrases)

    def generate_phrases(self, query_phrases: List[np.ndarray]) -> List[Tuple[int, np.ndarray]]:
        """
        Generate alapana phrases for the given set of query phrases.
        :param query_phrases: list of phrases as pitch contours performed by the lead artist
        :return: List of phrases to be performed by Hathaani
        """
        generated_phrases = []
        temp_silence = []
        self.used_idx = []
        for i, query in tqdm(enumerate(query_phrases), total=len(query_phrases)):
            hist = Util.compute_2d_histogram(query, self.bits)
            # progress = i * 1.0 / len(query_phrases)
            s0, x, s1 = self.__search(hist, query_progress=None, brute_force=False)

            if x is None:
                print(f"cluster empty. Skipping phrase {i} ...")
                continue

            s0 = int(s0 * self.max_silence_in_samples)
            s1 = int(s1 * self.max_silence_in_samples)

            if len(generated_phrases) == 0:
                silence = 0
            silence = 0 if len(temp_silence) == 0 else (s0 + temp_silence[-1]) // 2
            x_midi = Util.unnormalize_midi(x, self.target_midi_key)
            generated_phrases.append((silence, x_midi))
            temp_silence.append(s1)

        return generated_phrases

    def __search(self, query_histogram,
                 query_progress,
                 brute_force=False) -> Tuple[float, np.ndarray, float] or Tuple[None, None, None]:
        """
        Search the library for the most optimal phrase for the given query phrase
        :param query_histogram: Histogram of the query phrase
        :param query_progress: position of the phrase in alapana.
        If None, the algorithm uses loss function for the 2nd stage search
        :param brute_force: Whether to search using brute force or use kmeans clusters
        :return: The selected phrase
        """
        if brute_force:
            min_d = np.inf
            s0, x, s1 = self.library[0]
            for _s0, _x, _s1 in self.library:
                h = Util.compute_2d_histogram(_x, self.bits)
                d = self.loss_fn(query_histogram, h).sum()
                if min_d > d:
                    x = _x
                    s0 = _s0
                    s1 = _s1
                    min_d = d
            return s0, x, s1

        cluster = self.kmeans_model.get_cluster_for(query_histogram)
        cluster = np.setdiff1d(cluster, self.used_idx)

        candidates = []
        for i in cluster:
            candidates.append(self.library[i])

        if len(candidates) == 0:
            return None, None, None

        if query_progress is None:
            temp = []
            for _s0, _x, _s1 in candidates:
                h = Util.compute_2d_histogram(_x, self.bits)
                temp.append(h)
            candidates = np.array(temp)
            d = self.loss_fn(query_histogram, candidates)
            idx = cluster[np.argmin(d)]

            if idx in self.used_idx:
                print(f"Warn! {idx} repeated...")

            self.used_idx.append(idx)
            return self.library[idx]

        _s0, _x, _s1 = candidates[0]
        # running_min = np.inf
        # for _x, _s in candidates:
        #     temp = abs(query_progress - p)
        #     if temp < running_min:
        #         running_min = temp
        #         x = _x
        #         s = _s

        return _s0, _x, _s1

    def __preprocess_audio(self, audio: np.ndarray, fs: float, midi_key: int) -> List[np.ndarray]:
        # Split audio into phrase chunks
        bounds = Util.get_silence_bounds(audio, fs)
        audio_chunks = Util.split(audio, bounds, min_samples=int(fs * 1))
        # audio_chunks = audio_chunks[0:10]
        phrases = []
        for x, _ in tqdm(audio_chunks, total=len(audio_chunks)):
            p = Util.extract_pitch(x, fs, midi_key, self.pitch_tracker)
            phrases.append(p)

        return phrases

    # def __add_silences(self, phrases: List[np.ndarray]) -> List[Tuple[np.ndarray, int]]:
    #     """
    #     Append silence lengths in samples to each phrase in the list.
    #     :param phrases: List of phrases for which silence must be appended
    #     :return: List of tuple containing phrase and its corresponding silence length
    #     """
    #     print("Adding silences")
    #     out = []
    #     for i in tqdm(range(len(phrases) - 1)):
    #         current_phrase = torch.from_numpy(phrases[i]).reshape(1, -1, 1).to(self.device).float()
    #         next_phrase = torch.from_numpy(phrases[i + 1]).reshape(1, -1, 1).to(self.device).float()
    #         pred = self.silence_model(current_phrase=current_phrase, next_phrase=next_phrase)
    #         q_length = np.argmax(Util.to_numpy(pred))
    #         length = int(q_length * self.max_silence_in_samples / (self.divisions - 1))
    #         out.append((current_phrase.flatten(), length))
    #
    #     # there is no silence after the last phrase. So append 0 for length
    #     out.append((phrases[-1].flatten(), 0))
    #     return out
