import numpy as np


def cross_correlate(reference, surveillance):
    """
    Calculate the cross-correlation between reference
    and surveillance signals.

    Returns
    -------
    correlation : np.ndarray
        Cross-correlation result.
    lags : np.ndarray
        Corresponding sample lags.
    """

    correlation = np.correlate(
        surveillance,
        reference,
        mode="full",
    )

    lags = np.arange(
        -(len(reference) - 1),
        len(surveillance),
    )

    return correlation, lags