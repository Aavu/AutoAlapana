from alapana_nn.dataset import HistogramDataset
import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf
import alapana_nn.synth as synth
from alapana_nn.utils import Util
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from alapana_nn.pitchTrack import PitchTrack
from scipy.signal import spectrogram
from scipy.signal import medfilt
import time

# model = hub.load("https://tfhub.dev/google/spice/2").signatures["serving_default"]
tracker = PitchTrack('crepe', 50, 800, device='mps')
rmvpe_tracker = PitchTrack('rmvpe', 50, 800, device='cpu')
audio, fs = librosa.load("dataset/audio/train/1-48-6.wav", sr=16000)
audio = audio[:int(30 * fs)]
audio = audio / np.max(audio)
# sf.write("dataset/test/noisy_audio.wav", audio, samplerate=int(fs))

# output = model(tf.constant(audio, tf.float32))
# p = np.array(output["pitch"])
# confidence = np.array(1 - output["uncertainty"])

# start = time.time()
# _, f, conf, act = tracker.track(audio, fs)
# print(time.time() - start)
start = time.time()
f0 = rmvpe_tracker.track(audio, fs)
print(time.time() - start)
# t_out = librosa.hz_to_midi(f)
t0_out = librosa.hz_to_midi(f0)
# t_out[conf < 0.25] = np.nan


def output2hz(pitch_output):
    # Calibration constants
    PT_OFFSET = 25.58
    PT_SLOPE = 63.07
    FMIN = 10.0
    BINS_PER_OCTAVE = 12.0
    cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET
    return FMIN * 2.0 ** (1.0 * cqt_bin / BINS_PER_OCTAVE)


# p = librosa.hz_to_midi(output2hz(p))
# idx = np.argwhere(confidence < 0.1).flatten().astype(int)
# p[idx] = np.nan
# plt.plot(p)
# plt.subplot(211)
# plt.plot(t_out)
# plt.ylim([30, 80])
# plt.subplot(212)
plt.plot(t0_out)
plt.ylim([30, 80])
plt.show()
# data = HistogramDataset(root="dataset",
#                         train=True,
#                         divisions=None,
#                         device='cpu')
#
# fs = 44100
#
# for i, (s0, x, s1) in enumerate(data):
#     print(i)
#     x_midi = Util.unnormalize_midi(x[20:len(x) - 20], 62)
#     x_freq = librosa.midi_to_hz(x_midi)
#     x_audio = synth.simple_synth(x_freq, sr=fs)
#     sf.write("out.wav", x_audio, samplerate=fs)
#     plt.subplot(211)
#     plt.plot(x_midi)
#     plt.subplot(212)
#     plt.plot(np.diff(x_midi, 2))
#     plt.show()
