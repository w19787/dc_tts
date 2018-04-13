# encoding: utf-8
import tensorflow as tf
import functools
from hyperparams import Hyperparams as hp

"""
git hub: https://github.com/candlewill/Griffin_lim
TensorFlow implementation of Griffin-lim Algorithm for voice reconstruction
"""


# TF
def spectrogram2wav(spectrogram):
    '''Converts spectrogram into a waveform using Griffin-lim's raw.
    '''

    def invert_spectrogram(spectrogram):
        '''
        spectrogram: [t, f]
        '''
        spectrogram = tf.expand_dims(spectrogram, 0)
        inversed = tf.contrib.signal.inverse_stft(spectrogram, frame_length=hp.win_length, frame_step=hp.hop_length, window_fn=functools.partial(tf.contrib.signal.hann_window, periodic=True), fft_length=hp.n_fft)

        squeezed = tf.squeeze(inversed, 0)
        return squeezed

    # spectrogram = tf.transpose(spectrogram)

    spectrogram = tf.cast(spectrogram, dtype=tf.complex64)  # [t, f]
    X_best = tf.identity(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = tf.contrib.signal.stft(X_t, frame_length=hp.win_length, frame_step=hp.hop_length, fft_length=hp.n_fft, window_fn=functools.partial(tf.contrib.signal.hann_window, periodic=True), pad_end=False)  # (1, T, n_fft/2+1)
        phase = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)  # [t, f]
        X_best = spectrogram * phase  # [t, t]
    X_t = invert_spectrogram(X_best)
    y = tf.identity(tf.real(X_t), name='output')

    return y


def inv_spectrogram(spectrogram):
    S = _db_to_amp(_denormalize(spectrogram) + hp.ref_db)  # Convert back to linear
    # return _inv_preemphasis(spectrogram2wav(S ** 1.5))  # Reconstruct phase
    return spectrogram2wav(S ** 1.5)


def _denormalize(S):
    return (tf.clip_by_value(S, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db


def _db_to_amp(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def _inv_preemphasis(x):
    N = tf.shape(x)[0]
    i = tf.constant(0)
    W = tf.zeros(shape=tf.shape(x), dtype=tf.float32, name='output')

    def condition(i, y):
        return tf.less(i, N)

    def body(i, y):
        tmp = tf.slice(x, [0], [i + 1])
        tmp = tf.concat([tf.zeros([N - i - 1]), tmp], -1)
        y = hp.preemphasis * y + tmp
        i = tf.add(i, 1)
        return [i, y]

    final = tf.while_loop(condition, body, [i, W])

    y = final[1]

    return y