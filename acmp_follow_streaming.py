import os
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
from alapana_nn.synth import simple_synth, DDSPSynth
import soundfile as sf
from alapana_nn.utils import Util
from alapana_nn.pitchTrack import PitchTrack
from typing import List, Tuple
import sys

np.set_printoptions(precision=4, threshold=sys.maxsize)

pitch_threshold = 0.03
audio_threshold = 0.05
dev = "cpu"
hop = 160
CHUNK = 4096
fs = 16000
min_bow_length_ms = 150

x, _ = librosa.load("vocal_16k.wav", sr=fs)
x = x[:21 * fs]
audio_chunks = Util.split_chunks(x, CHUNK)

pitch_tracker = PitchTrack('rmvpe', hop_size=hop, rmvpe_threshold=pitch_threshold, device=dev)
ref_m = pitch_tracker.track(audio=x[:9 * fs], fs=fs, return_cents=True)
mid_x, mid_idx, sta, cpn = Util.decompose_carnatic_components(ref_m)
print(cpn)
# plt.scatter(sta, ref_m[sta])
# for note, s, l in cpn:
#     _x = np.arange(s, s + l + 1)
#     _y = np.ones_like(_x) * note
#     plt.plot(_x, _y, 'b--')
# ref_m[ref_m < 0.1] = np.nan
# plt.plot(ref_m)
# plt.plot(mid_idx, mid_x, 'r--')
# plt.ylim(40, 70)
# plt.show()

prev_x = np.zeros((CHUNK,), dtype=float)
# dummy run to initialize
cat_x = np.hstack([prev_x, prev_x])
zero_m = pitch_tracker.track(audio=cat_x, fs=fs, return_cents=True)
residual = zero_m[len(zero_m) // 2:3 * len(zero_m) // 4]

midi = []

last_sta = None
num_below_threshold = 0
for chunk in audio_chunks:
    e = Util.envelope(chunk, sample_rate=fs, hop_size=hop, normalize=False)
    cat_x = np.hstack([prev_x, chunk])
    m = pitch_tracker.track(audio=cat_x, fs=fs, return_cents=True)
    m[:len(m) // 4] = residual
    midi_chunk = m[: len(m) // 2]

    mid_x, mid_idx, sta, cpn = Util.decompose_carnatic_components(midi_chunk, first_sta_value=last_sta)
    peaks, dips, sil = Util.pick_stationary_points(midi_chunk)
    _sta = np.sort(np.hstack([peaks, dips]))

    # We dont know if the last sta idx is a true sta. So get the 2nd last
    last_sta = midi_chunk[_sta[-2]] if len(_sta) > 2 else None
    midi.extend(midi_chunk)

    residual = m[len(m) // 2:3 * len(m) // 4]
    prev_x = chunk

    if len(sta) > 0 or len(cpn) > 0:
        print(cpn)
        print(sta)
        print()
        # plt.scatter(_sta, midi_chunk[_sta], color='y')
        plt.scatter(sta, midi_chunk[sta])
        for note, s, l in cpn:
            _x = np.arange(s, s + l + 1)
            _y = np.ones_like(_x) * note
            plt.plot(_x, _y, 'b--')
        midi_chunk[midi_chunk < 0.1] = np.nan
        plt.plot(midi_chunk)
        plt.plot(mid_idx, mid_x, 'r--')
        plt.ylim(40, 70)
        plt.show()

    if e.mean() < audio_threshold:
        num_below_threshold += 1
        if num_below_threshold > 1:
            break

# midi = np.array(midi)[len(zero_m) // 4:]
plt.plot(ref_m + 5)
plt.show()
