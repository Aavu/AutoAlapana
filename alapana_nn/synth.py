import numpy as np
import librosa
from typing import Tuple
import torch
from omegaconf import OmegaConf
from alapana_nn.utils import Util
from ddsp.network.autoencoder import AutoEncoder


# refer: https://stackoverflow.com/questions/3089832/sine-wave-glissando-from-one-pitch-to-another-in-numpy
def simple_synth(_f0,
                 amplitude: int or np.ndarray = 0.25,
                 onsets=None,
                 f0_rate=100,
                 f0_hop=441,
                 sr=44100,
                 fade_ms=10) -> Tuple[np.ndarray, float]:
    f0_sr = f0_rate * f0_hop
    f = np.interp(np.arange(len(_f0) * f0_hop), np.arange(0, len(_f0) * f0_hop, f0_hop), _f0)
    end_time_sec = len(_f0) / f0_rate
    t = np.linspace(0, end_time_sec, int(np.round(f0_sr * end_time_sec)))
    df = np.diff(f, prepend=f[0])
    f[np.isnan(f)] = 0
    df[np.isnan(df)] = 0
    phi = np.cumsum(-t * 2 * np.pi * df)

    if type(amplitude) == np.ndarray:
        amplitude_sr = sr * len(amplitude) / len(t)
        amplitude = librosa.resample(amplitude, orig_sr=amplitude_sr, target_sr=sr, fix=True)
        assert len(t) == len(amplitude), f"len(t) = {len(t)}, len(amp) = {len(amplitude)}"

    _samples = amplitude * np.sin(2 * np.pi * f * t + phi)
    _samples[_samples == np.nan] = 0
    _samples = librosa.resample(y=_samples, orig_sr=f0_sr, target_sr=sr)

    if fade_ms > 0:
        fade_in_mask = np.linspace(0, 1, num=int(sr * fade_ms / 1000))
        fade_mask = np.ones_like(_samples)
        fade_mask[:len(fade_in_mask)] = fade_in_mask
        fade_mask = np.flip(fade_mask)
        fade_mask[:len(fade_in_mask)] = fade_in_mask
        _samples = _samples * fade_mask

    mask = np.ones_like(_samples)

    if onsets is not None:
        onsets = np.array(onsets)
        onsets = np.round(onsets * sr).astype(int)

        for i in range(1, len(onsets)):
            si = onsets[i] - (sr // 32)
            ei = onsets[i]
            mask[si: ei] = 0

        a = 1 - (1 / 100)
        mask = Util.zero_lpf(mask, a, restore_zeros=False)

    return _samples * mask, sr


class DDSPSynth:
    def __init__(self,
                 config_path="ddsp/configs/200220.pth.yaml",
                 ckpt_path="ddsp/weight/200220.pth",
                 hop_size=64,
                 sample_rate=16000,
                 device='cpu'):
        config = OmegaConf.load(config_path)
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.device = device
        self.model = AutoEncoder(config, hop_size=hop_size, sample_rate=sample_rate, device=device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))

    def synthesize(self,
                   f0: np.ndarray or torch.Tensor,
                   amplitude: np.ndarray or torch.Tensor,
                   normalize=True, include_reverb=True) -> np.ndarray or torch.Tensor:
        if type(f0) == np.ndarray:
            f0 = torch.from_numpy(f0.copy()).to(self.device)
        if type(amplitude) == np.ndarray:
            amplitude = torch.from_numpy(amplitude.copy()).to(self.device)

        return_attr = "audio_synth"
        if include_reverb:
            return_attr = "audio_reverb"
        samples = self.model.synth(f0.float(), amplitude.float(), normalize, return_attr=return_attr)
        samples = Util.to_numpy(samples)

        final_length = len(f0) * self.hop_size
        samples = samples[:final_length]

        return samples
