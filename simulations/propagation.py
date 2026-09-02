import numpy as np


def fractional_delay(signal, delay):
    """
    Apply a fractional sample delay to a complex signal
    using an FFT-based phase shift.

    Parameters
    ----------
    signal : np.ndarray
        Complex input signal.
    delay : float
        Delay in samples.

    Returns
    -------
    np.ndarray
        Delayed signal.
    """

    N = len(signal)

    pad = int(np.ceil(delay)) + 10

    padded = np.pad(
        signal,
        (pad, pad),
        mode="constant",
    )

    M = len(padded)

    spectrum = np.fft.fft(
        padded
    )

    frequencies = np.fft.fftfreq(
        M
    )

    phase_shift = np.exp(
        -1j
        * 2
        * np.pi
        * frequencies
        * delay
    )

    delayed = np.fft.ifft(
        spectrum
        * phase_shift
    )

    delayed = delayed[
        pad:pad + N
    ]

    return delayed


def apply_doppler(signal, doppler, sample_rate):
    """
    Apply a Doppler frequency shift to a complex baseband signal.

    Parameters
    ----------
    signal : np.ndarray
        Complex input signal.

    doppler : float
        Doppler frequency in Hz.

    sample_rate : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Doppler shifted signal.
    """

    n = np.arange(
        len(signal)
    )

    phase = np.exp(
        1j
        * 2
        * np.pi
        * doppler
        * n
        / sample_rate
    )

    return signal * phase