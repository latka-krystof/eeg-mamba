from torchvision.transforms import v2
from torchvision.transforms import functional as F
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

spectogram_transform = v2.Compose([torch.stft(), v2.Magnitude(), v2.AmplitudeToDB()])

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
        transform.append(spectogram_transform)
    # if mel_spectrogram:
    #     transform.append(v2.MelSpectrogram())
    return v2.Compose(transform)
