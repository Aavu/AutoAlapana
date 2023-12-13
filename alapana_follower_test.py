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

threshold = 0.03
dev = "cpu"
hop = 160
fs = 16000
min_bow_length_ms = 150

pitch_tracker = PitchTrack('rmvpe', hop_size=hop, rmvpe_threshold=threshold, device=dev)
synth = DDSPSynth(hop_size=hop, sample_rate=fs)

CHUNK = 2048
num_chunks = 33

x, _ = librosa.load("vocal_16k.wav", sr=fs)

# x = x[: CHUNK * (num_chunks + 1)]
# start_idx = int(0.644 * len(x))
# start_idx = int(0.09 * len(x))
# start_idx = int(0.19 * len(x))
# x = x[start_idx: start_idx + (25 * fs)]
silence_bounds = Util.get_silence_bounds(x, fs)
audio_phrases = Util.split(x, silence_bounds, min_samples=int(fs / 2))
# audio_phrases = [audio_phrases[12]]
final_audio = []

for i, (xx, sl) in enumerate(audio_phrases):
    p = pitch_tracker.track(audio=xx, fs=fs, return_cents=True)
    si, ei = Util.get_valid_boundaries(p)
    p = p[si: ei + 1]

    xx = xx[si * hop: (ei + 1) * hop]

    bins, _ = Util.pick_dips(xx, fs, hop, smoothing_alpha=0.9, wait_ms=min_bow_length_ms)
    filtered, idx, sta, cpn = Util.decompose_carnatic_components(p, threshold_semitones=0.3, min_cp_note_length=8)
    e = Util.envelope(xx, sample_rate=fs, hop_size=hop)[:len(p)]
    e = Util.zero_lpf(e, alpha=0.9, restore_zeros=False)
    mask = np.ones_like(e)
    bins = np.round(bins / (hop / fs)).astype(int)

    for _i in range(1, len(bins)):
        si = bins[_i] - (hop // 64)
        ei = bins[_i]
        mask[si: ei] = 0

    e = e * Util.zero_lpf(mask, 0.25, restore_zeros=False)

    _phrase = np.vstack([p, e]).T
    accompaniment = Util.generate_accompaniment_for_phrase(_phrase, cpn, sta, pitch_track_hop_size=hop)
    pitch, amp = accompaniment[:, 0], accompaniment[:, 1]

    # Add the original silence length from the singing
    length = len(p) + (sl // hop)

    max_length = max(length, len(pitch))
    p = np.concatenate([p, np.zeros((max_length - len(p),))])
    pitch = np.concatenate([pitch, np.zeros((max_length - len(pitch),))])
    amp = np.concatenate([amp, np.zeros((max_length - len(amp),))])

    x_freq = librosa.midi_to_hz(pitch) * 2
    x_audio = synth.synthesize(x_freq, amp)
    x_audio = x_audio * 0.25

    # xx = librosa.resample(xx, orig_sr=fs, target_sr=48000)

    if len(x_audio) > len(xx):
        xx = np.concatenate([xx, np.zeros((len(x_audio) - len(xx),))])
    else:
        x_audio = np.concatenate([x_audio, np.zeros((len(xx) - len(x_audio),))])

    final_audio.append(x_audio + xx)

    ################ Debug ################
    # out = x_audio + xx
    # sf.write(os.path.join("follow_test_audio", f"ddsp_sample.wav"), out, int(fs))
    # f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all', figsize=(15, 10), constrained_layout=True)
    # for note, s, l in cpn:
    #     x = np.arange(s, s + l)
    #     y = np.ones_like(x) * note
    #     ax1.plot(x, y, 'b--')
    # p[p < 0.1] = np.nan
    # pitch[pitch < 0.1] = np.nan
    # ax1.plot(p)
    # ax2.plot(pitch)
    # ax3.plot(amp)
    # plt.show()
    # print()
    ########################################

sf.write(os.path.join("follow_test_audio", f"ddsp.wav"), np.concatenate(final_audio), int(fs))
