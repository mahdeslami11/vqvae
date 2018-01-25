import tensorflow as tf
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


def mu_law(audio, quantization_channels=256):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)


def inv_mu_law(output, quantization_channels=256):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.to_float(output) / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu) ** abs(signal) - 1)
        return tf.sign(signal) * magnitude
