import torch
import librosa
from alapana_nn.pitchTrack import PitchTrack
from alapana_nn.synth import DDSPSynth
import soundfile as sf
import numpy as np


x, fs = librosa.load("vocal_16k.wav", sr=16000)
x = x[:16000 * 20]
print("File Loaded")

synth = DDSPSynth()
f0 = synth.model.get_f0(x) * 2
amp = synth.model.get_loudness(x)
out = synth.synthesize(f0, amp, include_reverb=True)
sf.write("out_synth1.wav", out, samplerate=int(fs))
