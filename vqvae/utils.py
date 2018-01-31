import numpy as np

def mu_law_numpy(x, quantization_channels=256):
    mu = quantization_channels - 1
    safe_audio_abs = np.minimum(np.abs(x), 1.0)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(x) * magnitude
    return ((signal + 1) / 2 * mu + 0.5).astype(np.int32)


def inv_mu_law_numpy(x, quantization_channels=256):
    mu = quantization_channels - 1
    signal = 2 * (x / mu) - 1
    magnitude = (1 / mu) * ((1 + mu) ** abs(signal) - 1)
    return np.sign(signal) * magnitude