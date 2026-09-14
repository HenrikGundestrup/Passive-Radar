
import numpy as np


def generate_reference_signal(N, seed=None):
    """
    Generate a complex Gaussian reference signal.
    """

    rng = np.random.default_rng(seed)

    return (
        rng.normal(size=N)
        + 1j * rng.normal(size=N)
    ) / np.sqrt(2)


def generate_noise(N, amplitude=1.0, seed=None):
    """
    Generate complex Gaussian receiver noise.
    """

    rng = np.random.default_rng(seed)

    return amplitude * (
        rng.normal(size=N)
        + 1j * rng.normal(size=N)
    ) / np.sqrt(2)


def generate_fm_signal(
    N,
    sample_rate,
    frequency_deviation=75e3,
):
    """
    Generate a synthetic audio-like FM baseband signal.

    The modulation consists of several audio-frequency
    components to create a more realistic FM waveform.
    """

    t = np.arange(N) / sample_rate

    # --------------------------------------------------------
    # Synthetic audio signal
    # --------------------------------------------------------

    audio = (
        0.60 * np.sin(2 * np.pi * 500 * t)
        + 0.30 * np.sin(2 * np.pi * 1200 * t)
        + 0.15 * np.sin(2 * np.pi * 2500 * t)
    )

    # Normalize audio
    audio /= np.max(np.abs(audio))

    # --------------------------------------------------------
    # FM modulation
    # --------------------------------------------------------

    instantaneous_frequency = (
        frequency_deviation * audio
    )

    phase = (
        2
        * np.pi
        * np.cumsum(instantaneous_frequency)
        / sample_rate
    )

    signal = np.exp(1j * phase)

    return signal

