import numpy as np


def fractional_delay(signal, delay):
    """
    Apply a fractional sample delay to a complex signal.

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
    n = np.arange(N)

    delayed_n = n - delay

    delayed_signal = (
        np.interp(
            delayed_n,
            n,
            signal.real,
            left=0,
            right=0,
        )
        + 1j
        * np.interp(
            delayed_n,
            n,
            signal.imag,
            left=0,
            right=0,
        )
    )

    return delayed_signal


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

    n = np.arange(len(signal))

    phase = np.exp(
        1j * 2 * np.pi * doppler * n / sample_rate
    )

    return signal * phase