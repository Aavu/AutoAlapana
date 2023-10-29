import numpy as np
import librosa


# refer: https://stackoverflow.com/questions/3089832/sine-wave-glissando-from-one-pitch-to-another-in-numpy
def simple_synth(_f0, f0_rate=100, hop=441, sr=44100, amplitude=0.25) -> np.ndarray:
    f0_sr = f0_rate * hop
    f = np.interp(np.arange(len(_f0) * hop), np.arange(0, len(_f0) * hop, hop), _f0)
    end_time_sec = len(_f0) / f0_rate
    t = np.linspace(0, end_time_sec, int(np.round(f0_sr * end_time_sec)))
    df = np.diff(f, prepend=f[0])
    phi = np.cumsum(-t * 2 * np.pi * df)
    _samples = amplitude * np.sin(2 * np.pi * f * t + phi)
    _samples = librosa.resample(y=_samples, orig_sr=f0_sr, target_sr=sr)
    return _samples
