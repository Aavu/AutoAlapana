import librosa
import numpy as np
from matplotlib import pyplot as plt
import sys
import alapana_nn.synth as synth
import soundfile as sf
from alapana_nn.generator import AlapanaGenerator

np.set_printoptions(precision=4, threshold=sys.maxsize)


if __name__ == "__main__":
    fs = 16000
    audio, _ = librosa.load("dataset/audio/test/1-51-5.wav", sr=fs)
    generator = AlapanaGenerator(pitch_detection_algorithm='rmvpe', target_midi_key=62)
    phrases = generator.generate_alapana_for(audio, fs, 51)
    out = []
    sr = fs
    for s, p in phrases:
        x_freq = librosa.midi_to_hz(p)
        x_audio, sr = synth.simple_synth(x_freq)
        silence = np.zeros((s,))
        out.append(silence)
        out.append(x_audio)
        print(s/fs)

    out = np.concatenate(out)
    sf.write("generated_silence4.wav", out, samplerate=sr)
