import numpy as np

from src.propagation import fractional_delay


def estimate_direct_path(
    reference,
    surveillance,
    expected_delay,
    search_width=20,
    fine_step=0.001,
):
    """
    Estimate the delay and complex amplitude of the direct-path signal.

    Parameters
    ----------
    reference : ndarray
        Transmitted/reference signal.

    surveillance : ndarray
        Received surveillance signal.

    expected_delay : float
        Expected direct-path delay in samples.

    search_width : int
        Integer search range around expected_delay.

    fine_step : float
        Resolution of the fractional-delay search.

    Returns
    -------
    estimated_delay : float
        Estimated direct-path delay in samples.

    coefficient : complex
        Estimated complex amplitude/phase coefficient.
    """

    # --------------------------------------------------------
    # Integer delay estimation
    # --------------------------------------------------------

    correlation = np.correlate(
        surveillance,
        reference,
        mode="full"
    )

    lags = np.arange(
        -(len(reference) - 1),
        len(surveillance)
    )

    positive = lags >= 0

    search_mask = (
        positive
        & (lags >= expected_delay - search_width)
        & (lags <= expected_delay + search_width)
    )

    search_indices = np.where(search_mask)[0]

    peak_index = search_indices[
        np.argmax(np.abs(correlation[search_indices]))
    ]

    integer_delay = int(lags[peak_index])

    # --------------------------------------------------------
    # Fine fractional-delay search
    # --------------------------------------------------------

    fine_delays = np.arange(
        integer_delay - 0.5,
        integer_delay + 0.5 + fine_step,
        fine_step
    )

    best_residual = np.inf
    best_delay = None
    best_coefficient = None

    for test_delay in fine_delays:

        test_reference = fractional_delay(
            reference,
            test_delay
        )

        denominator = np.vdot(
            test_reference,
            test_reference
        )

        if np.abs(denominator) < 1e-12:
            continue

        coefficient = (
            np.vdot(test_reference, surveillance)
            / denominator
        )

        estimated_direct = coefficient * test_reference

        residual = surveillance - estimated_direct

        residual_energy = np.vdot(
            residual,
            residual
        ).real

        if residual_energy < best_residual:
            best_residual = residual_energy
            best_delay = test_delay
            best_coefficient = coefficient

    return best_delay, best_coefficient


def cancel_direct_path(
    reference,
    surveillance,
    delay,
    coefficient,
):
    """
    Cancel the estimated direct-path signal.
    """

    estimated_direct = (
        coefficient
        * fractional_delay(reference, delay)
    )

    return surveillance - estimated_direct