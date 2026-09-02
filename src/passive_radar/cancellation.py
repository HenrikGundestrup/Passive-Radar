import numpy as np


def estimate_direct_path(reference, surveillance, delay):
    """
    Estimate the complex direct-path coefficient.

    Parameters
    ----------
    reference : np.ndarray
        Reference signal.
    surveillance : np.ndarray
        Surveillance signal.
    delay : int
        Estimated direct-path delay in samples.

    Returns
    -------
    complex
        Estimated complex amplitude coefficient.
    """

    delayed_reference = np.zeros_like(reference)

    if delay >= 0:
        delayed_reference[delay:] = reference[:-delay]
    else:
        delayed_reference[:delay] = reference[-delay:]

    denominator = np.vdot(
        delayed_reference,
        delayed_reference,
    )

    if np.abs(denominator) < 1e-12:
        return 0.0 + 0.0j

    coefficient = (
        np.vdot(
            delayed_reference,
            surveillance,
        )
        / denominator
    )

    return coefficient


def cancel_direct_path(
    reference,
    surveillance,
    delay,
    coefficient,
):
    """
    Cancel an estimated direct-path signal.

    Parameters
    ----------
    reference : np.ndarray
        Reference signal.
    surveillance : np.ndarray
        Surveillance signal.
    delay : int
        Direct-path delay in samples.
    coefficient : complex
        Estimated direct-path coefficient.

    Returns
    -------
    np.ndarray
        Surveillance signal after direct-path cancellation.
    """

    delayed_reference = np.zeros_like(reference)

    if delay >= 0:
        delayed_reference[delay:] = reference[:-delay]
    else:
        delayed_reference[:delay] = reference[-delay:]

    estimated_direct = coefficient * delayed_reference

    return surveillance - estimated_direct