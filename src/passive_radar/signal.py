import numpy as np


def generate_reference_signal(N, seed=None):
    """
    Generate a complex Gaussian reference signal.

    Parameters
    ----------
    N : int
        Number of samples.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Complex reference signal.
    """

    rng = np.random.default_rng(seed)

    return (
        rng.normal(size=N)
        + 1j * rng.normal(size=N)
    ) / np.sqrt(2)


def generate_noise(N, amplitude=1.0, seed=None):
    """
    Generate complex Gaussian receiver noise.

    Parameters
    ----------
    N : int
        Number of samples.
    amplitude : float
        Noise amplitude.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Complex noise signal.
    """

    rng = np.random.default_rng(seed)

    return amplitude * (
        rng.normal(size=N)
        + 1j * rng.normal(size=N)
    ) / np.sqrt(2)