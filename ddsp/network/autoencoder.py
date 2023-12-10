import torch
import torch.nn as nn
import librosa
from alapana_nn.pitchTrack import PitchTrack
from alapana_nn.utils import Util
from ddsp.components.harmonic_oscillator import HarmonicOscillator
from ddsp.components.reverb import TrainableFIRReverb
from ddsp.components.filtered_noise import FilteredNoise
from ddsp.network.decoder import Decoder
from ddsp.network.encoder import Encoder


class AutoEncoder(nn.Module):
    def __init__(self, config, hop_size, sample_rate, device='cpu'):
        """
        encoder_config
                use_z=False, 
                sample_rate=16000,
                z_units=16,
                n_fft=2048,
                hop_length=64,
                n_mels=128,
                n_mfcc=30,
                gru_units=512
        
        decoder_config
                mlp_units=512,
                mlp_layers=3,
                use_z=False,
                z_units=16,
                n_harmonics=101,
                n_freq=65,
                gru_units=512,

        components_config
                sample_rate
                hop_length
        """
        super().__init__()
        self.config = config
        self.device = device

        self.decoder = Decoder(config)
        self.encoder = Encoder(config)

        self.hop_size = hop_size
        self.config.sample_rate = sample_rate

        self.harmonic_oscillator = HarmonicOscillator(
            sr=config.sample_rate, frame_length=self.hop_size
        )

        self.filtered_noise = FilteredNoise(frame_length=self.hop_size)
        self.reverb = TrainableFIRReverb(reverb_length=config.sample_rate * 3)

        self.pitch_tracker = PitchTrack('rmvpe',
                                        rmvpe_threshold=config.f0_threshold,
                                        hop_size=self.hop_size,
                                        device=device)

    def forward(self, batch):
        """
        z

        input(dict(f0, z(optional), l)) : a dict object which contains key-values below
                f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, time)
                z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
                loudness : torch.tensor w/ shape(B, time)
        """
        batch = self.encoder(batch)
        latent = self.decoder(batch)

        harmonic = self.harmonic_oscillator(latent)
        noise = self.filtered_noise(latent)

        audio = dict(
            harmonic=harmonic, noise=noise, audio_synth=harmonic + noise[:, : harmonic.shape[-1]]
        )

        if self.config.use_reverb:
            audio["audio_reverb"] = self.reverb(audio)

        audio["a"] = latent["a"]
        audio["c"] = latent["c"]

        return audio

    def get_f0(self, x):
        """
        input:
            x = torch.tensor((1), wave sample)
        
        output:
            f0 : (n_frames, ). fundamental frequencies
        """
        self.eval()
        print(self.config.sample_rate)
        f0 = self.pitch_tracker.track(audio=x, fs=self.config.sample_rate, return_cents=False)
        f0[f0 < 0.3] = 0
        f0 = torch.from_numpy(f0).float().to(self.device)
        return f0

    def get_loudness(self, x):
        e = Util.envelope(x, sample_rate=self.config.sample_rate, hop_size=self.hop_size)
        e1 = Util.zero_lpf(e, alpha=0.9, restore_zeros=False)
        return e1

    def reconstruction(self, x, normalize=True):
        """
        input:
            x = torch.tensor((1), wave sample)

        output(dict):
            f0 : (n_frames, ). fundamental frequencies
            a : (n_frames, ). amplitudes
            c : (n_harmonics, n_frames). harmonic constants
            sig : (n_samples)
            audio_reverb : (n_samples + reverb, ). reconstructed signal
        """
        self.eval()

        print("Tracking Pitch")
        f0 = self.get_f0(x)

        print("Tracking Loudness")
        loudness = self.get_loudness(x)[:len(f0)]
        assert len(f0) == len(loudness), f"f0: {len(f0)}, loudness: {len(loudness)}"

        print("Generating Synth")
        return self.synth(f0, loudness, normalize=normalize, return_attr=None)

    def synth(self, f0: torch.Tensor,
              loudness: torch.Tensor,
              normalize=True,
              return_attr: str or None = "audio_reverb"):

        assert len(f0) == len(loudness), f"f0: {len(f0)}, loudness: {len(loudness)}"

        with torch.no_grad():
            batch = dict(f0=f0.unsqueeze(0), loudness=loudness.unsqueeze(0))
            recon = self.forward(batch)

            # make shape consistent(removing batch dim)
            for k, v in recon.items():
                val = v[0]
                if normalize:
                    val = val / torch.max(val)
                recon[k] = val

            recon["f0"] = f0
            recon["loudness"] = loudness

            if return_attr is not None:
                return recon[return_attr]

            return recon
