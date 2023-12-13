import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import torch
import librosa
from typing import List, Tuple
import os
import pickle
from scipy import signal
from alaapNet.data.transforms import ToTensor, FillGaps
from alapana_nn.pitchTrack import PitchTrack


class Util:
    @staticmethod
    def normalize_midi(pitches: ndarray, midi_key: int):
        out = 1 + (pitches - midi_key) / 12.0
        # out[out < 0] = 0
        # out[np.isnan(pitches)] = 0
        return out

    @staticmethod
    def unnormalize_midi(values, midi_key: int):
        return midi_key + ((values - 1) * 12)

    @staticmethod
    def pkl_load(filename: str):
        with open(filename, 'rb') as handle:
            return pickle.load(handle)

    @staticmethod
    def pkl_dump(filename: str, data):
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    def to_numpy(x: torch.Tensor):
        return x.cpu().detach().numpy()

    @staticmethod
    def js_div(x: ndarray, target: ndarray, eps=1e-6) -> float:
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
    def bit_crush(x: ndarray, bits, max_val):
        out = x[:]
        out = out / max_val
        # assert np.min(out) >= 0 and np.max(out) < 1, f"min = {np.min(out)}, max = {np.max(out)}"
        out *= 2 ** bits
        out = np.round(out).astype(int)
        out = np.maximum(0, np.minimum((2 ** bits) - 1, out))
        return out

    @staticmethod
    def histogram(x: ndarray, bins):
        out = np.zeros(bins, dtype=float)

        for i in x:
            if i >= bins:
                raise Exception(f"Values exceed bin count, i: {i}, bins: {bins}")
            out[i] += 1

        out = out / len(x)
        return out

    @staticmethod
    def fill_gaps(x):
        fg = FillGaps()
        x[np.abs(x) == np.inf] = 0
        x = fg(x)
        return x

    @staticmethod
    def compute_histogram(x, bits, max_val=3.0, fill_gaps=True):
        if fill_gaps:
            x = Util.fill_gaps(x)
        x_bit = Util.bit_crush(x, bits=bits, max_val=max_val)
        hist = Util.histogram(x_bit, bins=2 ** bits)
        return hist

    @staticmethod
    def compute_2d_histogram(x, bits, max_val=3.0, derivate_max_val=0.25, fill_gaps=True):
        """
        This function computes histogram for both the pitch contour and the 1st derivative
        :param x: pitch contour
        :param bits: bins = 2 ** bits
        :param max_val: maximum value of the pitch contour
        :param fill_gaps: Whether to fill gaps in pitch contour
        :return: 2d-array with histograms of x and diff(x). shape: (2, 2**bits)
        """
        if fill_gaps:
            x = Util.fill_gaps(x)
        h = Util.compute_histogram(x, bits, max_val, False)
        diff = np.diff(x)
        diff = (derivate_max_val + diff) / (2 * derivate_max_val)
        temp = np.clip(diff, a_min=0, a_max=1)
        hist = Util.compute_histogram(temp, bits, max_val=1, fill_gaps=False)
        return np.vstack([h.reshape(1, -1), hist.reshape(1, -1)])

    @staticmethod
    def get_silence_bounds(audio, fs, num_zeros=5, hop_sec=0.025, block_sec=0.05, eps=1e-6):
        hop = int(hop_sec * fs)
        e = librosa.feature.rms(y=audio, frame_length=int(block_sec * fs), hop_length=hop)[0]
        bounds = []
        i = 0
        while i < len(e) - num_zeros:
            if np.mean(e[i:i + num_zeros]) < eps:
                temp = i
                while np.mean(e[i:i + num_zeros]) < eps:
                    i += 1
                bounds.append((temp * hop, i * hop))
            else:
                i += 1

        return np.array(bounds)

    @staticmethod
    def split(audio: ndarray,
              silence_bounds: ndarray,
              min_samples: int) -> List[Tuple[ndarray, int]]:
        """
        Split the phrases in the audio. This function will split phrases concatenating the respective silence
        :param audio: audio data as numpy array
        :param silence_bounds: array of tuple containing start and stop indexes of silence
        :param min_samples: minimum number of samples to count as a phrase
        :return: list of phrases
        """

        def get_silence_length(idx):
            return silence_bounds[idx, 1] - silence_bounds[idx, 0]

        splits = []
        # starting
        t = audio[:silence_bounds[0, 0]]
        if len(t) > min_samples:
            splits.append((t, get_silence_length(0)))

        # All other bounds
        for i in range(len(silence_bounds) - 1):
            t = audio[silence_bounds[i, 1]: silence_bounds[i + 1, 0]]
            if len(t) > min_samples:
                splits.append((t, get_silence_length(i + 1)))

        # Ending
        t = audio[silence_bounds[-1, 1]:]
        if len(t) > min_samples:
            splits.append((t, get_silence_length(len(silence_bounds) - 1)))

        return splits

    @staticmethod
    def split_chunks(audio: ndarray, chunk_size: int):
        num_chunks = int(np.ceil(len(audio) / chunk_size))
        chunks = np.zeros((num_chunks, chunk_size))
        for i in range(len(chunks) - 1):
            chunks[i] = audio[i * chunk_size: (i + 1) * chunk_size]

        residual_idx = len(audio) - (len(audio) % chunk_size)
        if residual_idx <= len(audio):
            chunks[-1] = np.concatenate([audio[residual_idx:], np.zeros((chunk_size - (len(audio) - residual_idx)))])
        return chunks

    @staticmethod
    def extract_pitch(audio: ndarray, fs, midi_key, pitch_tracker: PitchTrack, threshold=0.01, fill_gaps=True):
        f0 = pitch_tracker.track(audio, fs)
        midi = librosa.hz_to_midi(f0)
        midi = Util.normalize_midi(midi, midi_key=midi_key)
        midi = Util.truncate_pitches(midi)
        if fill_gaps:
            fg = FillGaps()
            midi = fg(midi)
        return midi

    @staticmethod
    def truncate_pitches(midi):
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

    @staticmethod
    def zero_lpf(x: ndarray or torch.Tensor, alpha, restore_zeros=True, ignore_zeros=False):
        x0 = Util.lpf(x, alpha, restore_zeros=False, ignore_zeros=ignore_zeros)
        x0 = torch.flip(x0, dims=(0,)) if type(x) == torch.Tensor else np.flip(x0, axis=(0,))
        x0 = Util.lpf(x0, alpha, restore_zeros)
        x0 = torch.flip(x0, dims=(0,)) if type(x) == torch.Tensor else np.flip(x0, axis=(0,))
        return x0

    @staticmethod
    def lpf(x: ndarray or torch.Tensor, alpha, restore_zeros=True, ignore_zeros=False) -> ndarray or torch.Tensor:
        eps = 0.01
        if ignore_zeros:
            _x = Util.fill_zeros(x, eps=eps)
        else:
            _x = x.clone() if type(x) == torch.Tensor else x.copy()

        y = _x.clone() if type(_x) == torch.Tensor else _x.copy()

        for i in range(1, len(y)):
            y[i] = (alpha * y[i - 1]) + ((1 - alpha) * _x[i])

        # restore NULL values
        if restore_zeros:
            y[x < eps] = 0

        return y

    @staticmethod
    def segmented_zero_lpf(x: ndarray, alpha, restore_zeros=False):
        y = x.copy()
        seg = Util.split_phrases(x)
        for s, i in seg:
            temp = Util.zero_lpf(s, alpha, restore_zeros)
            y[i: i + len(s)] = temp
        return y

    @staticmethod
    def meanfilter(x: ndarray, kernel: int, ignore_zeros=False) -> ndarray:
        eps = 0.01
        _x = Util.fill_zeros(x, eps=eps) if ignore_zeros else x.copy()
        out = np.zeros_like(x)
        for i in range(kernel, len(x)):
            out[i] = np.mean(_x[i - kernel: i])

        return out

    @staticmethod
    def fill_zeros(x: ndarray or torch.Tensor, eps=0.01) -> ndarray or torch.Tensor:
        """
        This function is designed for causal system.
        This means, for example if the input is [0 1 1 1 1 0 0 0 0 0],
        the output will be [1 1 1 1 1 0 0 0 0 0] and not [1 1 1 1 1 1 1 1 1 1]
        :param x: input
        :param eps: threshold
        :return: filled array
        """
        if type(x) == torch.Tensor:
            _x = x.clone()
        else:
            _x = x.copy()

        for i in range(len(_x)):
            if _x[i] < eps:
                j = i + 1
                while j < len(_x) and _x[j] < eps:
                    j += 1
                if j >= len(_x):
                    break

                start = _x[i - 1] if i > 0 else _x[j]

                if type(_x) == torch.Tensor:
                    _x[i:j] = torch.linspace(start, _x[j], j - i)
                else:
                    _x[i:j] = np.linspace(start, _x[j], j - i)
        return _x

    @staticmethod
    def envelope(x: ndarray or torch.Tensor, sample_rate: float, hop_size: int, normalize: bool = True):
        eps = 1e-6
        # numpy and torch computations return different lengths! Usually len(np_env) - len(torch_env) = 1
        if type(x) == torch.Tensor:
            Zxx = torch.stft(x, n_fft=hop_size * 2, hop_length=hop_size, return_complex=True)
            env = torch.sum(torch.abs(Zxx), dim=0) + eps
            mav_env = torch.max(env)
        else:
            f, t, Zxx = signal.stft(x, sample_rate, nperseg=hop_size * 2, noverlap=hop_size)
            env = np.sum(np.abs(Zxx), axis=0) + eps
            mav_env = np.max(env)

        if normalize:
            env = env / mav_env

        return env

    @staticmethod
    def pick_dips(x: ndarray,
                  sample_rate: float = 16000,
                  hop_size: int = 160,
                  smoothing_alpha: float = 0.9,
                  wait_ms: int = 80) -> Tuple[ndarray, ndarray]:
        e = Util.envelope(x, sample_rate=sample_rate, hop_size=hop_size)
        lpf_e = Util.zero_lpf(e, alpha=smoothing_alpha, restore_zeros=False)
        wait_samples = (wait_ms / 1000) / (hop_size / sample_rate)

        dips = []
        entered_valley = False
        for i in range(1, len(e) - 1):
            if e[i] < lpf_e[i]:
                diff = np.diff(e[i - 1:i + 2])
                if diff[0] < diff[1]:
                    if not entered_valley:
                        dips.append([i])
                        entered_valley = True
                    else:
                        dips[-1].append(i)
            else:
                entered_valley = False

        for i in range(len(dips)):
            d = dips[i]
            dev = np.abs(lpf_e[d] - e[d])
            idx = np.argmax(dev)
            dips[i] = dips[i][idx]

        bins = dips[:]

        # rank and filter out until all dips are at least 'wait' distance apart
        def is_success():
            return (np.diff(bins) >= wait_samples).all() or len(bins) < 2

        while not is_success():
            dev = lpf_e[bins] - e[bins]
            min_idx = np.argmin(dev)

            del bins[min_idx]

        bins = np.array(bins) * (hop_size / sample_rate)
        return bins, lpf_e

    @staticmethod
    def pick_stationary_points(x: ndarray, eps: float = 0.1):
        peaks = []
        dips = []
        sil = []
        # move ptr to the first non zero value
        idx = 0
        while idx < len(x) - 1 and x[idx] < eps:
            sil.append(idx)
            idx += 1

        end_idx = len(x) - 1
        while end_idx > idx and x[end_idx] < eps:
            sil.append(end_idx)
            end_idx -= 1

        # Always make the first valid point as part of the sta. If 2nd pt > 1st, it is dip else peak
        if idx < len(x) - 1:
            if x[idx] < x[idx + 1]:
                dips.append(idx)
            else:
                peaks.append(idx)
            idx += 1

        # Always make the last valid point as part of the sta. If 2nd pt > 1st, it is dip else peak
        if end_idx > idx:
            if x[end_idx] < x[end_idx - 1]:
                dips.append(end_idx)
            else:
                peaks.append(end_idx)
            end_idx -= 1

        in_silent_region = False
        # Scan the entire contour. If a point is a minima, add to dip. If a point is maxima, add to peak
        for i in range(idx, end_idx):
            # First check if the next point is a valid pitch. If not, add current point to sta dip
            if x[i] < eps:
                if not in_silent_region:
                    dips.append(i - 1)
                    in_silent_region = True
                sil.append(i)
                continue

            # If still in silent region but the current point > eps, Add current point to sta
            if in_silent_region:
                if x[i + 1] > x[i]:
                    dips.append(i)
                else:
                    peaks.append(i)
                in_silent_region = False
                continue

            if x[i + 1] <= x[i] and x[i] > x[i - 1]:
                peaks.append(i)

            elif x[i + 1] > x[i] and x[i] <= x[i - 1]:
                dips.append(i)

        peaks = np.sort(np.array(peaks, dtype=int))
        dips = np.sort(np.array(dips, dtype=int))
        sil = np.sort(np.array(sil, dtype=int))
        return peaks, dips, sil

    @staticmethod
    def get_nearest_sta(sta_points: ndarray, idx: int, lower):
        # find the nearest sta that is before/after given idx.
        # If there is no sta, then original starting_idx is retained.
        if lower:
            for _i in range(len(sta_points) - 1, -1, -1):
                if sta_points[_i] <= idx:
                    idx = sta_points[_i]
                    break
        else:
            for _i in range(len(sta_points)):
                if sta_points[_i] >= idx:
                    idx = sta_points[_i]
                    break
        return idx

    @staticmethod
    def decompose_carnatic_components(x: ndarray,
                                      threshold_semitones: float = 0.3,
                                      min_cp_note_length: int = 10,
                                      min_sub_phrase_length: int = 3,
                                      first_sta_value: float = None):
        cpn = []
        peaks, dips, sil = Util.pick_stationary_points(x)
        sta = np.sort(np.hstack([peaks, dips]))
        if len(sta) < 2:
            return np.array([]), np.array([]), sta, cpn
        mid_x = np.zeros((len(sta) - 1,))
        mid_idx = np.zeros_like(mid_x)  # required only for plotting

        for i in range(len(mid_x)):
            if i == 0 and first_sta_value is not None:
                print(first_sta_value, x[sta[i]])
                mid_x[i] = (first_sta_value + x[sta[i + 1]]) / 2
            else:
                mid_x[i] = (x[sta[i]] + x[sta[i + 1]]) / 2
            mid_idx[i] = (sta[i] + sta[i + 1]) / 2

        eps = 0.1
        i = 0
        while i < len(mid_x):
            candidate_idx = None
            j = i + 1

            while j < len(mid_x) and abs(mid_x[j] - mid_x[j - 1]) < threshold_semitones and x[sta[j]] > eps and x[
                sta[j - 1]] > eps:
                if candidate_idx is None:
                    candidate_idx = j - 1
                j += 1

            if candidate_idx is not None:
                j1 = min(len(sta) - 1, j)
                s, e = sta[candidate_idx], sta[j1]
                l = e - s
                if l >= min_cp_note_length and len(sta[candidate_idx: j1]) > 2:
                    note = np.median(x[s: e])
                    appended = False
                    if len(cpn) > 0:
                        n0, s0, l0 = cpn[-1]
                        if (s + l) - (s0 + l0) < min_sub_phrase_length and abs(note - n0) < threshold_semitones:
                            cpn[-1] = ((n0 + note) / 2, s0, l0 + l)
                            appended = True

                    if not appended:
                        cpn.append((note, s, l))
                    sta[candidate_idx: j1] = -1
            i = j

        # handle last idx of sta
        if len(cpn) > 0:
            n0, s0, l0 = cpn[-1]
            if s0 <= sta[-1] <= s0 + l0:
                sta[-1] = -1

        sta = sta[sta > -1]
        return mid_x, mid_idx, sta, cpn

    @staticmethod
    def split_phrases(_x: np.ndarray, num_zeros: int = 5, min_phrase_length: int = 20, eps: float = 0.1):
        _phrases = []

        i = 0
        while i < len(_x):
            phrase_start_idx = None
            j = i + 1
            while j < len(_x) - 1 and _x[j] < eps: j += 1

            if phrase_start_idx is None:
                phrase_start_idx = j

            while j < len(_x) - 1 and _x[j] > eps: j += 1

            # If there is a potential phrase and the length of that phrase is > 0
            if phrase_start_idx is not None and j - phrase_start_idx > 0:
                # If the gap between the last phrase and this one is too small, it is probably the same phrase
                if phrase_start_idx - (i - 1) < num_zeros and len(_phrases) > 0:
                    # If the last phrase is too short, simply neglect it
                    if len(_phrases[-1][0]) < min_phrase_length:
                        _phrases[-1] = (_x[phrase_start_idx: j], phrase_start_idx)
                    else:
                        # include the gap in the phrase by filling with linear interpolation
                        itp = np.linspace(_x[i - 1], _x[phrase_start_idx], phrase_start_idx - (i - 1) - 1,
                                          endpoint=False)
                        _phrases[-1] = (
                            np.concatenate([_phrases[-1][0], itp, _x[phrase_start_idx: j]]), _phrases[-1][1])

                else:  # If the gap is larger, it probably means it's a new phrase
                    _phrases.append((_x[phrase_start_idx: j], phrase_start_idx))
            i = j

        return _phrases

    @staticmethod
    def get_valid_boundaries(x: ndarray, eps: float = 0.01) -> Tuple[int, int]:
        # Find the idx of first non-zero pitch
        _si = 0
        for _i in range(len(x)):
            _p = x[_i]
            if _p < eps or np.isnan(_p) or np.isinf(_p):
                continue

            _si = _i
            break

        # Find the idx of last non-zero pitch
        _ei = len(x) - 1
        for _i in range(_ei, _si - 1, -1):
            _p = x[_i]
            if _p < eps or np.isnan(_p) or np.isinf(_p):
                continue

            _ei = _i
            break

        return _si, _ei

    @staticmethod
    def model_periodic_signal(x: ndarray,
                              time_period,
                              max_freq: float = 20) -> Tuple[float, float, float]:
        """
        Given a periodic signal, estimate it's frequency, phase and amplitude
        :param x: periodic signal
        :param time_period: difference between t[1] and t[0]
        :param max_freq:
        :return: modelled sinusoidal parameters
        """
        peaks, dips, _ = Util.pick_stationary_points(x)
        # defaults
        f = 5.
        a = 0.1
        phi = 0

        if len(peaks) < 1 and len(dips) < 1:
            return f, a, phi

        if len(peaks) == 1 and len(dips) == 1:
            f0 = 1 / (abs(peaks[0] - dips[0]) * 2 * time_period)
            f = f0 if f0 < max_freq else f

        # There can be cases when there's only 1 peak but > 1 dips.
        # Use temp to add dummy peaks for f0 computation
        temp_peaks = list(peaks)
        if len(peaks) == 1 and len(dips) > 1:
            avg_dist = int(np.mean(np.diff(dips)))
            temp_peaks.append(peaks[-1] + avg_dist)

        # print(peaks, dips)
        # print(temp_peaks)

        # 1 pitch = h samples
        # 1 sample = 1/fs sec
        # 1 pitch = h/fs sec = 1 tp
        # t pitches = t * tp sec
        # f = 1 / t = 1 / (t * tp) hz = fs / (t * h)

        # h = 160, fs = 16000 -> t = 10.846153846153847, tp = 0.01
        # h = 128, fs = 16000 -> t = 13.538461538461538, tp = 0.008
        # h = 80, fs = 16000 -> t = 17.8, tp = 0.005
        # h = 64, fs = 16000 -> t = 17.526315789473685, tp = 0.004
        if len(temp_peaks) > 1:
            f0 = 1 / (np.mean(np.diff(temp_peaks)) * time_period)
            f = f0 if f0 < max_freq else f
        a = (np.mean(x[peaks]) - np.mean(x[dips])) / 2
        m = x[1] - x[0]
        t = 2 * np.pi * f * time_period
        # sin(t2 + phi) - sin(t1 + phi) = m
        sin_part = -2 * np.sin((-t / 2))
        if np.abs(m) <= np.abs(sin_part):
            phi = np.arccos(m / sin_part) - (t / 2)

        return f, a, phi

    @staticmethod
    def generate_lfo(note, length, time_period, freq, vibrato_amplitude, amplitude, phase):
        t = np.linspace(0, length * time_period, length)
        lfo = note + (vibrato_amplitude * np.sin(2 * np.pi * freq * t + phase)).reshape(-1, 1)
        _env = np.ones_like(lfo) * amplitude
        return np.concatenate([lfo, _env], axis=-1)

    @staticmethod
    def generate_cp_like(x: ndarray, note, time_period, amplitude, length):
        f, a, phi = Util.model_periodic_signal(x, time_period=time_period)
        f = f / 2  # this seems to be good for listening
        print(f"f: {f}, a: {a}, phi: {phi}")
        return Util.generate_lfo(note, length, time_period, f, a, amplitude, phi), f, a, phi

    @staticmethod
    def generate_accompaniment_for_phrase(phrase: np.ndarray,
                                          cp_notes: List[Tuple[float, int, int]],
                                          sta_pts: np.ndarray,
                                          pitch_track_hop_size: int = 160,
                                          sample_rate: float = 16000,
                                          max_len: int = 2500):
        """
        Generate accompaniment as pitch contour for a single phrase given its corresponding cp_notes
        :param phrase: Pitch contour and loudness curve of the phrase. Shape (N, 2)
        :param cp_notes: Constant-Pitch notes of the phrase
        :param sta_pts: The stationary points of the phrase
        :param pitch_track_hop_size: Hop size used during pitch tracking
        :param sample_rate: Audio sample rate
        :param max_len: Maximum possible length of accompaniment.
        :return: Accompaniment as a pitch contour with the same hop size as the input
        """

        # hyperparameters
        full_phrase_repeat_length_sec = 2.5
        AMPLITUDE_DURING_SINGING = 0.15
        min_cp_length_sec = 0.75
        min_sub_phrase_length_sec = 0.5
        follow_phrase_length_sec = 2.45
        max_accompaniment_response_sec = 0.25
        fade_length_sec = 0.5
        subphrase_to_cp_ratio = 6

        # The minimum length of the accompaniment is the length of phrase
        acmp = np.zeros((max_len, 2))

        assert full_phrase_repeat_length_sec > follow_phrase_length_sec

        def add_fade(_phrase: np.ndarray, length_sec, fade_out=False):
            out = _phrase.copy()
            fade_length = min(int(length_sec / seconds_per_pitch), len(_phrase) // 2)
            fade_mask = np.linspace(0, 1., fade_length)
            fade_out_mask = np.array([])
            if fade_out:
                fade_out_mask = np.flip(fade_mask)

            _mask = np.concatenate([fade_mask,
                                    np.ones((len(_phrase) - len(fade_mask) - len(fade_out_mask),)),
                                    fade_out_mask])

            if len(_phrase.shape) == 2:
                out[:, 1] = _phrase[:, 1] * _mask
            elif len(_phrase.shape) == 1:
                out = _phrase * _mask
            else:
                raise RuntimeError("Num dimensions can be at most be 2...")
            return out

        # each sample in pitch is hop samples (160). each second is fs samples (16000)
        # sec / pitch = (samples / pitch) / (samples / sec)
        seconds_per_pitch = pitch_track_hop_size / sample_rate  # 160 / 16000 = 0.01
        max_response_fill_length = int(max_accompaniment_response_sec / seconds_per_pitch)
        assert max_response_fill_length > 0

        follow_phrase_length = follow_phrase_length_sec / seconds_per_pitch  # 1 / 0.01 = 100 pitch samples per sec
        full_phrase_repeat_length = int(full_phrase_repeat_length_sec / seconds_per_pitch)

        # if a phrase is short, play it completely.
        if len(phrase) < full_phrase_repeat_length:
            phrase = add_fade(phrase, fade_length_sec / 4, fade_out=True)
            return np.concatenate([np.zeros((len(phrase), 2)), phrase], axis=0)

        # print(f"Phrase length ({len(phrase)}) > phrase repeat length ({full_phrase_repeat_length})")

        print(f"raw cp: {cp_notes}")
        # filter cp_notes and keep only cp larger than min_cp_length
        min_cp_length = int(min_cp_length_sec / seconds_per_pitch)
        filtered_cp = []
        for i, c in enumerate(cp_notes):
            if c[2] >= min_cp_length:  # or i == len(cp_notes) - 1:
                _a, _b, _c = c
                filtered_cp.append((c[0], c[1], c[2]))
        cp_notes = filtered_cp

        # If there are no cp notes after filtering and the phrase is longer,
        # play the last n seconds of the phrase
        if len(cp_notes) == 0:
            # Play the last 100 pitches
            # Start from the nearest sta to make sure the starting velocity is 0
            starting_idx = (len(phrase) - 1) - int(follow_phrase_length)
            starting_idx = Util.get_nearest_sta(sta_pts, starting_idx, lower=True)
            temp = add_fade(phrase[starting_idx:], fade_length_sec / 2, fade_out=True)
            return np.concatenate([np.zeros((len(phrase), 2)), temp],
                                  axis=0)

        # If both the above cases are not satisfied, there is at least 1 cp-note that is long enough
        # Get all phrases inbetween cp. Including before and after the first and last cp respectively

        # Append phrase before the first cp
        bw_phrase_idx = [(0, cp_notes[0][1])]

        for i in range(len(cp_notes) - 1):
            _s = cp_notes[i][1] + cp_notes[i][2]
            _e = cp_notes[i + 1][1]
            bw_phrase_idx.append((_s, _e))

        cp = cp_notes[-1]

        min_sub_phrase_length = int(min_sub_phrase_length_sec / seconds_per_pitch)
        _s, _e = cp[1] + cp[2], len(phrase)

        # If the last sub-phase is too short, it's most likely a tracking glitch. So include it in the cp-note.
        if _e - _s < min_sub_phrase_length:
            cp_notes[-1] = (cp[0], cp[1], cp[2] + (_e - _s))
            bw_phrase_idx.append((_s, _s))
        else:
            bw_phrase_idx.append((_s, _e))

        print(f"cp notes: {cp_notes}")
        print(f"b/w phrase indices: {bw_phrase_idx}")

        # This ptr is used to move to the right index in the acmp output when filling in values
        ptr = max_response_fill_length
        for i in range(len(cp_notes)):
            # i'th phrase in bw_phrases comes before i'th cp.
            cp = cp_notes[i]
            _s, _e = bw_phrase_idx[i]
            sub_phrase_length = _e - _s
            cp_start_idx = cp[1]
            cp_end_idx = cp[1] + cp[2]
            is_last = cp_end_idx >= len(phrase) - 1

            # play the last cp during sub-phrase singing
            if i > 0:
                prev_cp = cp_notes[i - 1]
                temp, f, a, phi = Util.generate_cp_like(phrase[prev_cp[1]: prev_cp[1] + prev_cp[2], 0],
                                                        prev_cp[0], seconds_per_pitch,
                                                        AMPLITUDE_DURING_SINGING, sub_phrase_length)
                adj_ratio = (-8 * (a ** 2)) + (2 * a) + 10
                # print(f"Adjusted ratio: {adj_ratio}, a: {a}")
                if sub_phrase_length / prev_cp[2] < adj_ratio:
                    acmp[ptr: ptr + sub_phrase_length] = temp
            ptr += sub_phrase_length

            sub_phrase = np.zeros((0, 2))
            temp_cp = np.zeros((0, 2))

            # We need to keep playing until the cp-note is held.
            # If the sub-phrase is too short, dont play the sub-phrase. Just play the cp-note starting from sub-phrase
            if sub_phrase_length < min_sub_phrase_length:
                temp_cp = phrase[_s: cp_end_idx]
                if acmp[ptr - 1, 0] < 0.1:
                    temp_cp = add_fade(temp_cp, fade_length_sec)
                print("here1")

            # if sub-phrase is shorter than cp-note length or if this cp extends till the last idx,
            # play the full sub-phrase
            elif sub_phrase_length <= cp[2] or is_last:
                _s = Util.get_nearest_sta(sta_pts, _s, lower=True)
                sub_phrase = phrase[_s: _e]
                _end_idx = len(phrase) if is_last else cp_end_idx - len(sub_phrase)
                temp_cp = phrase[cp_start_idx: _end_idx]
                if acmp[ptr - 1, 0] < 0.1:
                    sub_phrase = add_fade(sub_phrase, fade_length_sec)
                print("here2")

            # Since cp-note is shorter, play only the last n seconds of the sub-phrase.
            # But if sub-phrase is too long for cp, just play the cp
            else:
                _start_idx = _e - cp[2] if sub_phrase_length / cp[2] < subphrase_to_cp_ratio else cp_start_idx
                _start_idx = Util.get_nearest_sta(sta_pts, _start_idx, lower=True)
                print(_start_idx, cp_end_idx)
                sub_phrase = phrase[_start_idx: cp_end_idx]
                if acmp[ptr - 1, 0] < 0.1:
                    sub_phrase = add_fade(sub_phrase, fade_length_sec)
                print("here3")

            acmp[ptr: ptr + len(sub_phrase)] = sub_phrase
            ptr += len(sub_phrase)
            acmp[ptr: ptr + len(temp_cp)] = temp_cp
            ptr += len(temp_cp)

        # handle last phrase if any
        _s, _e = bw_phrase_idx[-1]
        sub_phrase_length = _e - _s
        cp_length = _e - ptr
        if sub_phrase_length > min_sub_phrase_length and cp_length > 0:
            prev_cp = cp_notes[-1]
            temp, f, a, phi = Util.generate_cp_like(phrase[prev_cp[1]: prev_cp[1] + prev_cp[2], 0],
                                                    prev_cp[0], seconds_per_pitch,
                                                    AMPLITUDE_DURING_SINGING, cp_length)
            adj_ratio = (-8 * (a ** 2)) + (2 * a) + 10
            # print(f"Adjusted ratio: {adj_ratio}, a: {a}")
            if cp_length / prev_cp[2] < adj_ratio:
                acmp[ptr: ptr + cp_length] = temp
                ptr += cp_length

        ptr = max(ptr, len(phrase) - 1)

        acmp = acmp[:ptr]
        if sub_phrase_length > min_sub_phrase_length:
            _s = Util.get_nearest_sta(sta_pts, _s, lower=True)
            sub_phrase = phrase[_s: _e]
            # Add fade ins
            if acmp[ptr - 1, 0] < 0.1:
                sub_phrase = add_fade(sub_phrase, fade_length_sec)
            acmp = np.concatenate([acmp, sub_phrase], axis=0)

        acmp[:, 0] = Util.segmented_zero_lpf(acmp[:, 0], 0.4)
        acmp[:, 1] = Util.zero_lpf(acmp[:, 1], 0.5, restore_zeros=False)
        return acmp
