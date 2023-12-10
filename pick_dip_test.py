import librosa
import numpy as np
from alapana_nn.synth import simple_synth
import soundfile as sf
from alapana_nn.utils import Util
from alapana_nn.pitchTrack import PitchTrack

threshold = 0.03
dev = "cpu"
hop = 160
min_bow_length_ms = 80

x, fs = librosa.load("vocal_16k.wav", sr=16000)
# x = x[0 * int(fs):1 * 60 * int(fs)]
x = x[: 16045]
bins, lpe = Util.pick_dips(x, fs, hop, smoothing_alpha=0.9, wait_ms=min_bow_length_ms)
print(bins)

pitch_tracker = PitchTrack('rmvpe', rmvpe_threshold=threshold, device=dev)
f0 = pitch_tracker.track(x, fs, return_cents=False)

lpe = lpe / np.max(lpe)
print(f0.shape, lpe.shape, x.shape)

samples, sr = simple_synth(f0*2, onsets=bins, amplitude=lpe)
# sf.write("test1.wav", samples, int(sr), format='wav')
