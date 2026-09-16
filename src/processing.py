import numpy as np


def cross_correlate(reference, surveillance):
    """
    Calculate the full cross-correlation between
    the reference and surveillance signals.
    """

    correlation = np.correlate(
        surveillance,
        reference,
        mode="full"
    )

    lags = np.arange(
        -(len(reference) - 1),
        len(surveillance)
    )

    return correlation, lags


def range_doppler_processing(
    reference,
    surveillance,
    sample_rate,
    block_size,
    num_blocks,
    direct_path_delay,
):
    """
    Perform range-Doppler processing using block-based
    cross-correlation followed by a Doppler FFT.

    Only delays at or beyond the direct-path delay
    are retained.

    Returns
    -------
    doppler_map : ndarray
        Complex range-Doppler map.

    delay_axis : ndarray
        Excess delay axis in samples, relative to
        the direct path.

    doppler_axis : ndarray
        Doppler axis in Hz.
    """

    fft_size = 2 * block_size - 1

    delay_profiles = []

    # --------------------------------------------------------
    # Block-based delay processing
    # --------------------------------------------------------

    for block in range(num_blocks):

        start = block * block_size
        end = start + block_size

        reference_block = reference[start:end]

        surveillance_block = surveillance[start:end]

        reference_fft = np.fft.fft(
            reference_block,
            fft_size
        )

        surveillance_fft = np.fft.fft(
            surveillance_block,
            fft_size
        )

        correlation = np.fft.ifft(
            surveillance_fft
            * np.conj(reference_fft)
        )

        # Rearrange correlation so that negative delays
        # appear before positive delays.
        correlation = np.concatenate(
            (
                correlation[-(block_size - 1):],
                correlation[:block_size]
            )
        )

        delay_profiles.append(correlation)

    delay_profiles = np.array(
        delay_profiles
    )

    # --------------------------------------------------------
    # Original delay axis
    # --------------------------------------------------------

    delay_axis_full = np.arange(
        -(block_size - 1),
        block_size
    )

    # --------------------------------------------------------
    # Remove delays before direct path
    # --------------------------------------------------------

    direct_delay_bin = int(
        round(direct_path_delay)
    )

    valid = delay_axis_full >= direct_delay_bin

    delay_profiles = delay_profiles[:, valid]

    delay_axis = (
        delay_axis_full[valid]
        - direct_delay_bin
    )

    # --------------------------------------------------------
    # Doppler processing
    # --------------------------------------------------------

    doppler_map = np.fft.fftshift(
        np.fft.fft(
            delay_profiles,
            axis=0
        ),
        axes=0
    )

    # --------------------------------------------------------
    # Doppler axis
    # --------------------------------------------------------

    doppler_axis = np.fft.fftshift(
        np.fft.fftfreq(
            num_blocks,
            d=block_size / sample_rate
        )
    )

    return (
        doppler_map,
        delay_axis,
        doppler_axis
    )


def detect_peak(
    doppler_map,
    delay_axis,
    doppler_axis,
):
    """
    Find the strongest peak in the range-Doppler map.

    Returns
    -------
    detected_delay : int
        Detected excess delay in samples.

    detected_doppler : float
        Detected Doppler frequency in Hz.

    peak_power : float
        Power of the strongest peak in the range-Doppler map.
    """

    # --------------------------------------------------------
    # Calculate power
    # --------------------------------------------------------

    power = np.abs(doppler_map) ** 2

    # --------------------------------------------------------
    # Find strongest peak
    # --------------------------------------------------------

    peak_index = np.unravel_index(
        np.argmax(power),
        power.shape
    )

    doppler_index = peak_index[0]
    delay_index = peak_index[1]

    # --------------------------------------------------------
    # Convert indices to physical quantities
    # --------------------------------------------------------

    detected_delay = delay_axis[
        delay_index
    ]

    detected_doppler = doppler_axis[
        doppler_index
    ]

    peak_power = power[
        peak_index
    ]

    return (
        detected_delay,
        detected_doppler,
        peak_power
    )