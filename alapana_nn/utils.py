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
            y[i: i + len(s)] = Util.zero_lpf(s, alpha, restore_zeros)
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
        # numpy and torch computations return different lengths! Usually len(np_env) - len(torch_env) = 1
        if type(x) == torch.Tensor:
            Zxx = torch.stft(x, n_fft=hop_size * 2, hop_length=hop_size, return_complex=True)
            env = torch.sum(torch.abs(Zxx), dim=0)
            mav_env = torch.max(env)
        else:
            f, t, Zxx = signal.stft(x, sample_rate, nperseg=hop_size * 2, noverlap=hop_size)
            env = np.sum(np.abs(Zxx), axis=0)
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
        idx = 1
        while idx < len(x) - 1 and x[idx] < eps:
            sil.append(idx)
            idx += 1

        # Always make the first point as part of the sta. If 2nd pt > 1st, it is dip else peak
        if idx < len(x) - 2:
            if x[idx] < x[idx + 1]:
                dips.append(idx)
            else:
                peaks.append(idx)
            idx += 1

        in_silent_region = False
        # Scan the entire contour. If a point is a minima, add to dip. If a point is maxima, add to peak
        for i in range(idx, len(x) - 1):
            # First check if the next point is a valid pitch. If not, add this point to sta dip
            if x[i] < eps:
                if not in_silent_region:
                    dips.append(i - 1)
                in_silent_region = True
                sil.append(i)
                continue

            if in_silent_region:
                if x[i + 1] > x[i]:
                    dips.append(i)
                else:
                    peaks.append(i)
                in_silent_region = False
                continue

            if x[i + 1] <= x[i] and x[i] > x[i - 1]:
                peaks.append(i)

            if x[i + 1] > x[i] and x[i] <= x[i - 1]:
                dips.append(i)

        peaks = np.array(peaks, dtype=int)
        dips = np.array(dips, dtype=int)
        sil = np.array(sil, dtype=int)
        return peaks, dips, sil

    @staticmethod
    def decompose_carnatic_components(x: ndarray,
                                      sample_rate: float = 16000,
                                      hop_size: int = 160,
                                      threshold_semitones: float = 0.3,
                                      min_cp_note_length_ms: int = 80) -> Tuple[
        ndarray, ndarray, ndarray, List[Tuple[float, int, int]]]:
        x = x.copy()
        note_length = int((min_cp_note_length_ms / 1000) / (hop_size / sample_rate))

        cpn = []
        eps = 0.1
        peaks, dips, sil = Util.pick_stationary_points(x)
        raw_sta = np.sort(np.hstack([peaks, dips, sil]))
        mid_idx = np.zeros_like(raw_sta)
        mid_x = np.zeros_like(raw_sta, dtype=float)
        mid_idx[0] = raw_sta[0]
        mid_x[0] = x[raw_sta[0]]

        for i in range(1, len(raw_sta)):
            mid_idx[i] = (raw_sta[i] + raw_sta[i - 1]) // 2
            mid_x[i] = (x[raw_sta[i]] + x[raw_sta[i - 1]]) / 2

        i = 0
        while i < len(mid_x):
            candidate_cpn_idx = None
            j = i + 1

            while j < len(mid_x) \
                    and abs(mid_x[j] - mid_x[j - 1]) < threshold_semitones \
                    and mid_x[j] > eps and mid_x[j - 1] > eps:

                if candidate_cpn_idx is None:
                    candidate_cpn_idx = (mid_idx[j] + mid_idx[j - 1]) // 2

                j += 1

            if candidate_cpn_idx is not None:

                end_idx = (mid_idx[j - 1] + mid_idx[min(j, len(mid_x) - 1)]) // 2
                s, e = candidate_cpn_idx, end_idx
                l = e - s
                if l >= note_length:
                    note = np.median(x[candidate_cpn_idx: end_idx])
                    cpn.append((note, s, e))

            i = j

        if len(cpn) == 0:
            return mid_x, mid_idx, raw_sta, cpn
        # now filter raw sta's by removing sta's that are part of cpn
        sta = []

        internal_eps = threshold_semitones * 1.5

        # Handle sta's before the first cpn
        n, s, e = cpn[0]
        sta.extend(list(raw_sta[raw_sta < s]))

        for i in range(len(cpn) - 1):
            n, s1, e = cpn[i]
            n1, s, e1 = cpn[i + 1]

            temp = raw_sta[raw_sta > e]
            sta.extend(list(temp[temp < s]))
            cpn[i] = (n, s1, e)
            cpn[i + 1] = (n1, s, e1)

        # Handle sta's after the last cpn
        n, s, e = cpn[-1]
        # e = expand(int(e), n, internal_eps, stop=len(x))
        # cpn[-1] = (n, s, e)
        sta.extend(list(raw_sta[raw_sta > e]))

        final_cpn = [cpn[0]]
        for i in range(1, len(cpn)):
            n, s, e = cpn[i]
            n1, s1, e1 = final_cpn[-1]

            if abs(n - n1) < internal_eps:
                if s - e1 <= 2:
                    final_cpn[-1] = ((n + n1) / 2, s1, e)
                    continue
            final_cpn.append(cpn[i])

        # convert end idx to lengths in cpn
        for i in range(len(final_cpn)):
            n, s, e = final_cpn[i]
            final_cpn[i] = (n, s, (e - s))

        return mid_x, mid_idx, np.array(sta), final_cpn

    @staticmethod
    def split_phrases(_x: np.ndarray, num_zeros: int = 5, min_phrase_length: int = 20, eps: float = 0.1):
        _phrases = []

        last_cp_idx = 0
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
                if phrase_start_idx - (i - 1) < num_zeros:
                    # If the last phrase is too short, simply neglect it
                    if len(_phrases[-1][0]) < min_phrase_length:
                        _phrases[-1] = (_x[phrase_start_idx: j], phrase_start_idx)
                    else:
                        # include the gap in the phrase by filling with linear interpolation
                        itp = np.linspace(_x[i - 1], _x[phrase_start_idx], phrase_start_idx - (i - 1), endpoint=False)
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
                              max_freq: float = 14) -> Tuple[float, float, float]:
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
            f0 = abs(peaks[0] - dips[0]) / 2
            f = f0 if f0 < max_freq else f

        # There can be cases when there's only 1 peak but > 1 dips.
        # Use temp to add dummy peaks for f0 computation
        temp_peaks = list(peaks)
        if len(peaks) == 1 and len(dips) > 1:
            avg_dist = int(np.mean(np.diff(dips)))
            temp_peaks.append(peaks[-1] + avg_dist)

        # print(peaks, dips)
        # print(temp_peaks)

        if len(temp_peaks) > 1:
            f0 = np.mean(np.diff(temp_peaks))
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
    def generate_lfo(note, length, hop, freq, vibrato_amplitude, amplitude, phase):
        t = np.linspace(0, length / hop, length)
        lfo = note + (vibrato_amplitude * np.sin(2 * np.pi * freq * t + phase)).reshape(-1, 1)
        _env = np.ones_like(lfo) * amplitude
        return np.concatenate([lfo, _env], axis=-1)

    @staticmethod
    def generate_cp_like(x: ndarray, note, hop, amplitude, length):
        f, a, phi = Util.model_periodic_signal(x, time_period=(1 / hop))
        print(f"f: {f}, a: {a}, phi: {phi}")
        return Util.generate_lfo(note, length, hop, f, a, amplitude, phi)
