import time
import numpy as np
import matplotlib.pyplot as plt

from src.geometry import (
    C,
    bistatic_range,
    bistatic_delay,
    bistatic_doppler,
)

from src.radar_signal import (
    generate_fm_signal,
    generate_noise,
)

from src.propagation import (
    fractional_delay,
    apply_doppler,
)

from src.cancellation import (
    estimate_direct_path,
    cancel_direct_path,
)

from src.processing import (
    range_doppler_processing,
    detect_peak,
)


def run_simulation(
    sample_rate=1e6,
    block_size=2048,
    num_blocks=128,
    carrier_frequency=100e6,
    target_amplitude=0.02,
    direct_path_amplitude=10.0,
    noise_amplitude=0.5,
    noise_seed=123,
    show_plot=True,
):
    """
    Run one complete passive radar simulation.

    Parameters
    ----------
    sample_rate : float
        Sampling frequency in Hz.

    block_size : int
        Number of samples in each processing block.

    num_blocks : int
        Number of processing blocks.

    carrier_frequency : float
        Carrier frequency in Hz.

    target_amplitude : float
        Amplitude of the target-reflected signal.

    direct_path_amplitude : float
        Amplitude of the direct-path signal.

    noise_amplitude : float
        Amplitude of receiver noise.

    noise_seed : int or None
        Random seed for the noise generator.

    show_plot : bool
        If True, display the Range-Doppler map.

    Returns
    -------
    dict
        Dictionary containing simulation results.
    """

    # ========================================================
    # Parameters
    # ========================================================

    N = block_size * num_blocks

    # ========================================================
    # Geometry
    # ========================================================

    tx_position = np.array([0.0, 0.0])
    rx_position = np.array([10_000.0, 0.0])

    target_position = np.array([5_000.0, 20_000.0])
    target_velocity = np.array([0.0, 250.0])

    tx_range = np.linalg.norm(
        target_position - tx_position
    )

    rx_range = np.linalg.norm(
        target_position - rx_position
    )

    true_target_range = bistatic_range(
        tx_position,
        target_position,
        rx_position,
    )

    true_target_delay = bistatic_delay(
        tx_position,
        target_position,
        rx_position,
    )

    true_target_delay_samples = (
        true_target_delay * sample_rate
    )

    direct_path_range = np.linalg.norm(
        rx_position - tx_position
    )

    direct_path_delay_samples = (
        direct_path_range / C * sample_rate
    )

    true_excess_delay_samples = (
        true_target_delay_samples
        - direct_path_delay_samples
    )

    doppler = bistatic_doppler(
        tx_position,
        target_position,
        rx_position,
        target_velocity,
        carrier_frequency,
    )

    wavelength = C / carrier_frequency

    # ========================================================
    # Generate FM reference signal
    # ========================================================

    reference = generate_fm_signal(
        N=N,
        sample_rate=sample_rate,
        frequency_deviation=75e3,
    )

    # ========================================================
    # Generate target signal
    # ========================================================

    target = fractional_delay(
        reference,
        true_target_delay_samples,
    )

    target *= target_amplitude

    target = apply_doppler(
        target,
        doppler,
        sample_rate,
    )

    # ========================================================
    # Generate direct path
    # ========================================================

    direct_path = fractional_delay(
        reference,
        direct_path_delay_samples,
    )

    direct_path *= direct_path_amplitude

    # ========================================================
    # Generate noise
    # ========================================================

    noise = generate_noise(
        N,
        amplitude=noise_amplitude,
        seed=noise_seed,
    )

    target_power = np.mean(
        np.abs(target) ** 2
    )

    noise_power = np.mean(
        np.abs(noise) ** 2
    )

    if target_power == 0:
        snr_db = -np.inf

    else:
        snr_db = 10 * np.log10(
            target_power / noise_power
        )

    # ========================================================
    # Surveillance signal
    # ========================================================

    surveillance = (
        target
        + direct_path
        + noise
    )

    # ========================================================
    # Automatic direct-path estimation
    # ========================================================

    start_time = time.perf_counter()

    estimated_direct_delay, direct_coefficient = (
        estimate_direct_path(
            reference,
            surveillance,
            direct_path_delay_samples,
        )
    )

    direct_path_time = time.perf_counter() - start_time

    # ========================================================
    # Direct-path cancellation
    # ========================================================

    surveillance_cancelled = cancel_direct_path(
        reference,
        surveillance,
        estimated_direct_delay,
        direct_coefficient,
    )

    # ========================================================
    # Range-Doppler processing
    # ========================================================

    start_time = time.perf_counter()

    doppler_map, delay_axis, doppler_axis = (
        range_doppler_processing(
            reference,
            surveillance_cancelled,
            sample_rate,
            block_size,
            num_blocks,
            direct_path_delay_samples,
        )
    )

    range_doppler_time = time.perf_counter() - start_time

    # ========================================================
    # Target detection
    # ========================================================

    detected_delay, detected_doppler, peak_power = detect_peak(
        doppler_map,
        delay_axis,
        doppler_axis,
    )

    # ========================================================
    # Plot
    # ========================================================

    if show_plot:

        power_db = (
            20
            * np.log10(
                np.abs(doppler_map)
                / np.max(np.abs(doppler_map))
                + 1e-12
            )
        )

        plt.figure(figsize=(10, 6))

        plt.imshow(
            power_db,
            aspect="auto",
            origin="lower",
            extent=[
                delay_axis[0],
                delay_axis[-1],
                doppler_axis[0],
                doppler_axis[-1],
            ],
        )

        plt.colorbar(
            label="Relative power [dB]"
        )

        plt.xlabel("Excess delay [samples]")
        plt.ylabel("Doppler [Hz]")

        plt.title(
            "Passive Radar Range-Doppler Map"
        )

        plt.tight_layout()
        plt.show()

    # ========================================================
    # Return results
    # ========================================================

    return {
        "tx_range": tx_range,
        "rx_range": rx_range, 
        "true_target_range": true_target_range,
        "true_target_delay": true_target_delay,
        "true_target_delay_samples": true_target_delay_samples,
        "direct_path_range": direct_path_range,
        "direct_path_delay_samples": direct_path_delay_samples,
        "true_excess_delay_samples": true_excess_delay_samples,
        "doppler": doppler,
        "wavelength": wavelength,
        "estimated_direct_delay": estimated_direct_delay,
        "direct_coefficient": direct_coefficient,
        "detected_delay": detected_delay,
        "detected_doppler": detected_doppler,
        "doppler_map": doppler_map,
        "delay_axis": delay_axis,
        "doppler_axis": doppler_axis,
        "direct_path_time": direct_path_time,
        "range_doppler_time": range_doppler_time,
        "target_power": target_power,
        "noise_power": noise_power,
        "snr_db": snr_db,
        "peak_power": peak_power,
    }


if __name__ == "__main__":

    # ========================================================
    # Run simulation
    # ========================================================

    result = run_simulation()

    # ========================================================
    # Print geometry
    # ========================================================

    print()
    print("=" * 60)
    print("BISTATIC RADAR GEOMETRY")
    print("=" * 60)

    print(
        f"TX → Target range:   "
        f"{result['tx_range']:.2f} m"
    )

    print(
        f"Target → RX range:   "
        f"{result['rx_range']:.2f} m"
    )

    print(
        f"Bistatic range:      "
        f"{result['true_target_range']:.2f} m"
    )

    print()
    print(
        f"Direct path range:   "
        f"{result['direct_path_range']:.2f} m"
    )

    print(
        f"Direct path delay:   "
        f"{result['direct_path_delay_samples']:.3f} samples"
    )

    print()
    print(
        f"Propagation time:    "
        f"{result['true_target_delay'] * 1e6:.3f} µs"
    )

    print(
        f"Target delay:        "
        f"{result['true_target_delay_samples']:.3f} samples"
    )

    print()
    print(
        f"Target velocity:     "
        f"{250.0:.2f} m/s"
    )

    print(
        f"Carrier frequency:   "
        f"{100e6:.2f} Hz"
    )

    print(
        f"Wavelength:          "
        f"{result['wavelength']:.3f} m"
    )

    print(
        f"Doppler shift:       "
        f"{result['doppler']:.2f} Hz"
    )

    # ========================================================
    # Direct-path estimation
    # ========================================================

    print()
    print("=" * 60)
    print("AUTOMATIC DIRECT-PATH ESTIMATION")
    print("=" * 60)

    print(
        f"True direct delay:       "
        f"{result['direct_path_delay_samples']:.3f} samples"
    )

    print(
        f"Estimated direct delay:  "
        f"{result['estimated_direct_delay']:.3f} samples"
    )

    print(
        f"True direct amplitude:   "
        f"{10.0:.6f}"
    )

    print(
        f"Estimated amplitude:     "
        f"{abs(result['direct_coefficient']):.6f}"
    )

    print(
        f"Estimated phase:         "
        f"{np.angle(result['direct_coefficient']):.6f} rad"
    )

    # ========================================================
    # Detection
    # ========================================================

    print()
    print("=" * 60)
    print("PASSIVE RADAR RANGE-DOPPLER")
    print("=" * 60)

    print(
        f"True delay:       "
        f"{result['true_excess_delay_samples']:.3f} samples"
    )

    print(
        f"Detected delay:   "
        f"{result['detected_delay']} samples"
    )

    print()

    print(
        f"True Doppler:     "
        f"{result['doppler']:.2f} Hz"
    )

    print(
        f"Detected Doppler: "
        f"{result['detected_doppler']:.2f} Hz"
    )

    print()

    print(
        f"Target amplitude:      "
        f"{0.02}"
    )

    print(
        f"Direct path amplitude: "
        f"{10.0}"
    )

    print(
        f"Noise amplitude:       "
        f"{0.5}"
    )