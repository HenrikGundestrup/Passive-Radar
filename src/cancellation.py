import numpy as np

from src.propagation import fractional_delay


def estimate_direct_path(
    reference,
    surveillance,
    expected_delay,
    search_width=20,
):
    """
    Estimate the delay and complex amplitude of the direct-path signal.

    The delay is estimated in two steps:

    1. Find the integer-sample correlation peak.
    2. Estimate the fractional part using parabolic interpolation
       around the correlation peak.

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

    Returns
    -------
    estimated_delay : float
        Estimated direct-path delay in samples.

    coefficient : complex
        Estimated complex amplitude/phase coefficient.
    """

    # ========================================================
    # FFT-based cross-correlation
    # ========================================================

    N = len(reference)
    M = len(surveillance)

    fft_length = N + M - 1

    reference_fft = np.fft.fft(
        reference,
        fft_length,
    )

    surveillance_fft = np.fft.fft(
        surveillance,
        fft_length,
    )

    correlation_fft = np.fft.ifft(
        surveillance_fft
        * np.conj(reference_fft)
    )

    # Rearrange circular correlation into the same lag
    # ordering as np.correlate(..., mode="full").
    correlation = np.concatenate(
        (
            correlation_fft[-(N - 1):],
            correlation_fft[:M],
        )
    )

    lags = np.arange(
        -(N - 1),
        M,
    )

    # ========================================================
    # Search around expected direct-path delay
    # ========================================================

    search_mask = (
        (lags >= expected_delay - search_width)
        & (lags <= expected_delay + search_width)
    )

    search_indices = np.where(search_mask)[0]

    peak_index = search_indices[
        np.argmax(
            np.abs(
                correlation[search_indices]
            )
        )
    ]

    integer_delay = int(
        lags[peak_index]
    )

    # ========================================================
    # Fractional-delay estimation
    # ========================================================

    # Make sure we have neighbours on both sides.
    if (
        peak_index <= 0
        or peak_index >= len(correlation) - 1
    ):
        fractional_offset = 0.0

    else:

        y_left = np.abs(
            correlation[peak_index - 1]
        )

        y_center = np.abs(
            correlation[peak_index]
        )

        y_right = np.abs(
            correlation[peak_index + 1]
        )

        denominator = (
            y_left
            - 2.0 * y_center
            + y_right
        )

        if abs(denominator) < 1e-12:
            fractional_offset = 0.0

        else:
            fractional_offset = (
                0.5
                * (y_left - y_right)
                / denominator
            )

    estimated_delay = (
        integer_delay
        + fractional_offset
    )

    # ========================================================
    # Estimate complex amplitude
    # ========================================================

    estimated_reference = fractional_delay(
        reference,
        estimated_delay,
    )

    denominator = np.vdot(
        estimated_reference,
        estimated_reference,
    )

    if abs(denominator) < 1e-12:
        return estimated_delay, 0.0 + 0.0j

    coefficient = (
        np.vdot(
            estimated_reference,
            surveillance,
        )
        / denominator
    )

    return estimated_delay, coefficient


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
        * fractional_delay(
            reference,
            delay,
        )
    )

    return surveillance - estimated_direct