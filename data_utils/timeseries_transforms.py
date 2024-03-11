from torchvision.transforms import v2
from torchvision.transforms import functional as F
import torchaudio.transforms as T
import torch


class Jittering(object):
    def __init__(self, mean, sigma=0.3):
        self.sigma = sigma
        self.mean = mean

    def __call__(self, x):
        return x + torch.normal(self.mean, self.sigma, size=x.size())


class GaussianNoise(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return x + torch.randn_like(x)


class Scaling(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        return x * self.scale


class Interpolate(object):
    def __init__(self, size, mode):
        self.size = size
        self.mode = mode

    def __call__(self, x):
        return F.interpolate(x, size=self.size, mode=self.mode)


# For simplicity
class FFT(object):
    def __call__(
        self,
        x,
    ):
        return torch.fft.fft(x)

class Trimming(object):
    def __init__(self, length=500):
        self.length = length

    def __call__(self, x):
        return x[:, 500-self.length//2:500+self.length//2]

class LowPassFilter(object):
    def __init__(self, cutoff_freq=0.1):
        self.cutoff_freq = cutoff_freq

    def __call__(self, x):
        frequency_domain = torch.fft.fft(x)
        n = x.size(-1)
        frequency_domain[:, int(n * self.cutoff_freq) : int(n * (1 - self.cutoff_freq))] = 0
        return torch.fft.ifft(frequency_domain).real

class Spectrogram(object):

    def __init__(self, n_fft=256, win_length=64, hop_length=16, window_fn=torch.hamming_window):
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_fn = window_fn

        self.spec = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fn,
        )

    def __call__(
        self,
        x,
    ):
        x = x + torch.randn_like(x)
        # x = Trimming()(x)
        # x = LowPassFilter()(x)
        spectrogram = self.spec(x)
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()
        return spectrogram

def create_transform(
    jittering=False,
    GaussianNoise=False,
    scaling=False,
    interpolate=False,
    spectrogram=False,
    mel_spectrogram=False,
    mean=0,
    scale=1,
    size=128,
    mode="linear",
):
    transform = []
    if jittering:
        transform.append(Jittering(mean))
    if GaussianNoise:
        transform.append(GaussianNoise())
    if scaling:
        transform.append(Scaling(scale))
    if interpolate:
        transform.append(Interpolate(size, mode))
    if spectrogram:
        transform.append(Spectrogram())
    # if mel_spectrogram:
    #     transform.append(v2.MelSpectrogram())
    return v2.Compose(transform)
