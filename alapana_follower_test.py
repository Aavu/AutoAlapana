import os.path
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
    full_phrase_repeat_length_sec = 2
    AMPLITUDE_DURING_SINGING = 0.15
    min_cp_length_sec = 0.3
    min_sub_phrase_length_sec = 0.5
    follow_phrase_length_sec = 1.5
    max_accompaniment_response_sec = 0.25
    fade_length_sec = 0.5
    subphrase_to_cp_ratio = 6

    # The minimum length of the accompaniment is the length of phrase
    acmp = np.zeros((max_len, 2))

    assert full_phrase_repeat_length_sec > follow_phrase_length_sec

    # each sample in pitch is hop samples (160). each second is fs samples (16000)
    # sec / pitch = (samples / pitch) / (samples / sec)
    seconds_per_pitch = pitch_track_hop_size / sample_rate  # 160 / 16000 = 0.01
    max_response_fill_length = int(max_accompaniment_response_sec / seconds_per_pitch)
    assert max_response_fill_length > 0

    def get_nearest_sta(_idx, lower):
        # find the nearest sta that is before starting idx.
        # If there is no sta, then original starting_idx is retained.
        if lower:
            for _i in range(len(sta_pts) - 1, -1, -1):
                if sta_pts[_i] <= _idx:
                    _idx = sta_pts[_i]
                    break
        else:
            for _i in range(len(sta_pts)):
                if sta_pts[_i] >= _idx:
                    _idx = sta_pts[_i]
                    break
        return _idx

    follow_phrase_length = follow_phrase_length_sec / seconds_per_pitch  # 1 / 0.01 = 100 pitch samples per sec
    full_phrase_repeat_length = int(full_phrase_repeat_length_sec / seconds_per_pitch)

    # if a phrase is short, play it completely.
    if len(phrase) < full_phrase_repeat_length:
        return np.concatenate([np.zeros((len(phrase), 2)), phrase], axis=0)

    print(f"Phrase length ({len(phrase)}) > phrase repeat length ({full_phrase_repeat_length})")

    # filter cp_notes and keep only cp larger than min_cp_length
    min_cp_length = int(min_cp_length_sec / seconds_per_pitch)
    filtered_cp = []
    for c in cp_notes:
        if c[2] >= min_cp_length:
            _a, _b, _c = c
            filtered_cp.append((c[0], c[1], c[2]))
    cp_notes = filtered_cp

    # If there are no cp notes after filtering and the phrase is longer,
    # play the last n seconds of the phrase
    if len(cp_notes) == 0:
        # Play the last 100 pitches
        # Start from the nearest sta to make sure the starting velocity is 0
        starting_idx = (len(phrase) - 1) - int(follow_phrase_length)
        # starting_idx = get_nearest_sta(starting_idx, lower=True)
        return np.concatenate([np.zeros((len(phrase) + max_response_fill_length, 2)), phrase[starting_idx:]], axis=0)

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

    def add_fade(_phrase: np.ndarray, length_sec):
        out = _phrase.copy()
        fade_mask = np.linspace(0, 1., int(length_sec / seconds_per_pitch))
        _mask = np.concatenate([fade_mask, np.ones_like(_phrase[:, 1])])
        out[:, 1] = _phrase[:, 1] * _mask[:len(_phrase)]
        return out

    # This ptr is used to move to the right index in the acmp output when filling in values
    ptr = max_response_fill_length
    for i in range(len(cp_notes)):
        # i'th phrase in bw_phrases comes before i'th cp.
        cp = cp_notes[i]
        _s, _e = bw_phrase_idx[i]
        _start_idx = cp[1]
        _end_idx = cp[1] + cp[2]
        is_last = _end_idx >= len(phrase) - 1

        # play the last cp during sub-phrase singing
        if i > 0:
            prev_cp = cp_notes[i - 1]
            if (_e - _s) / prev_cp[2] < subphrase_to_cp_ratio:
                acmp[ptr: ptr + (_e - _s)] = Util.generate_cp_like(phrase[prev_cp[1]: prev_cp[1] + prev_cp[2], 0],
                                                                   prev_cp[0], hop, AMPLITUDE_DURING_SINGING, _e - _s)
        ptr += (_e - _s)

        sub_phrase = np.zeros((0, 2))
        temp_cp = np.zeros((0, 2))

        # We need to keep playing until the cp-note is held.
        # If the sub-phrase is too short, dont play the sub-phrase. Just play the cp-note starting from sub-phrase
        if _e - _s < min_sub_phrase_length:
            temp_cp = phrase[_s: _end_idx]
            if acmp[ptr - 1, 0] < 0.1:
                temp_cp = add_fade(temp_cp, fade_length_sec)
            print("here1")

        # if sub-phrase is shorter than cp-note length or if this cp extends till the last idx, play the full sub-phrase
        elif _e - _s <= cp[2] or is_last:
            sub_phrase = phrase[_s: _e]
            _end_idx = len(phrase) if is_last else _end_idx - len(sub_phrase)
            temp_cp = phrase[_start_idx: _end_idx]
            if acmp[ptr - 1, 0] < 0.1:
                sub_phrase = add_fade(sub_phrase, fade_length_sec)
            print("here2")

        # Since cp-note is shorter, play only the last n seconds of the sub-phrase.
        # But if sub-phrase is too long for cp, just play the cp
        else:
            _start_idx = _e - cp[2] if (_e - _s) / cp[2] < subphrase_to_cp_ratio else _start_idx
            sub_phrase = phrase[_start_idx: _start_idx + cp[2]]
            if acmp[ptr - 1, 0] < 0.1:
                sub_phrase = add_fade(sub_phrase, fade_length_sec)
            print("here3")

        acmp[ptr: ptr + len(sub_phrase)] = sub_phrase
        ptr += len(sub_phrase)
        acmp[ptr: ptr + len(temp_cp)] = temp_cp
        ptr += len(temp_cp)

    # handle last phrase if any
    _s, _e = bw_phrase_idx[-1]
    if _e - _s > min_sub_phrase_length:
        prev_cp = cp_notes[-1]
        if (_e - _s) / prev_cp[2] < subphrase_to_cp_ratio:
            acmp[ptr: ptr + (_e - _s)] = Util.generate_cp_like(phrase[prev_cp[1]: prev_cp[1] + prev_cp[2], 0],
                                                               prev_cp[0], hop, AMPLITUDE_DURING_SINGING, _e - _s)
            ptr += (_e - _s)

    ptr = max(ptr, len(phrase) - 1)

    acmp = acmp[:ptr]
    if _e - _s > min_sub_phrase_length:
        # Add fade ins
        sub_phrase = phrase[_s: _e]
        if acmp[ptr - 1, 0] < 0.1:
            sub_phrase = add_fade(sub_phrase, fade_length_sec)
        acmp = np.concatenate([acmp, sub_phrase], axis=0)

    acmp[:, 0] = Util.segmented_zero_lpf(acmp[:, 0], 0.5)
    acmp[:, 1] = Util.zero_lpf(acmp[:, 1], 0.5, restore_zeros=False)
    return acmp


threshold = 0.03
dev = "cpu"
hop = 160
fs = 16000
min_bow_length_ms = 150

pitch_tracker = PitchTrack('rmvpe', hop_size=hop, rmvpe_threshold=threshold, device=dev)
synth = DDSPSynth(hop_size=hop, sample_rate=fs)

CHUNK = 2048
num_chunks = 300

x, _ = librosa.load("vocal_16k.wav", sr=fs)
# x = x[: CHUNK * (num_chunks + 1)]  # 65536
# start_idx = int(0.644 * len(x))
# start_idx = int(0.09 * len(x))
# x = x[start_idx: start_idx + (20 * fs)]
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
    filtered, idx, sta, cpn = Util.decompose_carnatic_components(p,
                                                                 sample_rate=fs,
                                                                 hop_size=hop,
                                                                 threshold_semitones=0.3,
                                                                 min_cp_note_length_ms=80)
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
    accompaniment = generate_accompaniment_for_phrase(_phrase, cpn, sta, pitch_track_hop_size=hop)
    pitch, amp = accompaniment[:, 0], accompaniment[:, 1]

    length = len(p) + (sl // hop)

    max_length = max(length, len(pitch))
    p = np.concatenate([p, np.zeros((max_length - len(p),))])
    pitch = np.concatenate([pitch, np.zeros((max_length - len(pitch),))])
    amp = np.concatenate([amp, np.zeros((max_length - len(amp),))])

    x_freq = librosa.midi_to_hz(pitch)
    x_audio = synth.synthesize(x_freq, amp)
    x_audio = x_audio * 0.25

    if len(x_audio) > len(xx):
        xx = np.concatenate([xx, np.zeros((len(x_audio) - len(xx),))])
    else:
        x_audio = np.concatenate([x_audio, np.zeros((len(xx) - len(x_audio),))])

    final_audio.append(x_audio + xx)

    ################ Debug ################
    # out = x_audio + xx
    # print(f"sample length: {len(out)}")
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
    #########################################
    print()

sf.write(os.path.join("follow_test_audio", f"ddsp.wav"), np.concatenate(final_audio), int(fs))
