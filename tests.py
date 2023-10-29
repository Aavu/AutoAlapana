import librosa
import matplotlib.pyplot as plt
import numpy as np
from rmvpe import rmvpe
from synth import simple_synth
import soundfile as sf

# x, fs = librosa.load("sample3.wav", sr=None)
# stft = np.abs(librosa.stft(y=x, win_length=2048, hop_length=2048))
# e = np.sum(stft, axis=0)
# # fig, ax = plt.subplots()
# # img = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max), y_axis='log', x_axis='time',ax=ax)
# # fig.colorbar(img, ax=ax, format="%+2.0f dB")
# e = librosa.amplitude_to_db(e)
# e = e / np.max(e)
# inv_e = 1 - e
# peaks = librosa.util.peak_pick(inv_e, pre_max=5, post_max=5, pre_avg=5, post_avg=5, delta=0.05, wait=5)
# for p in peaks:
#     plt.axvline(x=p, color='r')
# plt.plot(e)
# plt.show()

x, fs = librosa.load("dataset/audio/1-50-4.wav", sr=16000)
model_path = "rmvpe/rmvpe.pt"
threshold = 0.03
dev = "mps"

model = rmvpe.RMVPE(model_path, is_half=False, device=dev)
f0 = model.infer_from_audio(x[fs*60*2:fs*60*3], thred=threshold)
midi = librosa.hz_to_midi(f0)
midi = 1 + (midi - 51) / 12.0
# samples = simple_synth(f0*2)
# sf.write("test.wav", samples, 44100, format='wav')
plt.plot(midi)
plt.show()
