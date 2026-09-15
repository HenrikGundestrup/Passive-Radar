
import time

from simulations.basic_simulation import run_simulation


def main():

    start_time = time.perf_counter()

    number_of_trials = 100

    target_amplitudes = [
        0.001,
        0.002,
        0.003,
        0.005,
        0.0075,
        0.010,
        0.015,
        0.020,
    ]

    delay_tolerance = 2
    doppler_tolerance = 10.0

    results = []

    print()
    print("=" * 70)
    print("MONTE CARLO DETECTION EXPERIMENT")
    print("=" * 70)

    print(
        f"Trials per amplitude: {number_of_trials}"
    )

    print(
        f"Delay tolerance:      ±{delay_tolerance} samples"
    )

    print(
        f"Doppler tolerance:    ±{doppler_tolerance:.1f} Hz"
    )

    print()

    # ========================================================
    # Monte Carlo experiment
    # ========================================================

    for target_amplitude in target_amplitudes:

        detections = 0
        snr_values = []

        for trial in range(number_of_trials):

            result = run_simulation(
                show_plot=False,
                noise_seed=trial,
                target_amplitude=target_amplitude,
            )

            # ------------------------------------------------
            # SNR
            # ------------------------------------------------

            snr_values.append(
                result["snr_db"]
            )

            # ------------------------------------------------
            # Detection error
            # ------------------------------------------------

            delay_error = abs(
                result["detected_delay"]
                - result["true_excess_delay_samples"]
            )

            doppler_error = abs(
                result["detected_doppler"]
                - result["doppler"]
            )

            detected = (
                delay_error <= delay_tolerance
                and
                doppler_error <= doppler_tolerance
            )

            if detected:
                detections += 1

        # ----------------------------------------------------
        # Average SNR over all trials
        # ----------------------------------------------------

        average_snr = (
            sum(snr_values)
            / len(snr_values)
        )

        detection_probability = (
            detections / number_of_trials
        )

        results.append(
            (
                target_amplitude,
                average_snr,
                detections,
                detection_probability,
            )
        )

        print(
            f"Amplitude {target_amplitude:.4f} | "
            f"SNR {average_snr:6.2f} dB | "
            f"Detections {detections}/{number_of_trials} | "
            f"Probability {detection_probability * 100:5.1f} %"
        )

    total_time = time.perf_counter() - start_time

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"{'Target amplitude':>18} "
        f"{'SNR [dB]':>12} "
        f"{'Detections':>14} "
        f"{'Probability':>14}"
    )

    print("-" * 70)

    for (
        target_amplitude,
        average_snr,
        detections,
        detection_probability,
    ) in results:

        print(
            f"{target_amplitude:18.4f} "
            f"{average_snr:12.2f} "
            f"{detections:>7d}/{number_of_trials:<6d} "
            f"{detection_probability * 100:>10.1f} %"
        )

    print("-" * 70)

    print(
        f"Total experiment time: "
        f"{total_time:.2f} s"
    )

    print(
        f"Average time per trial: "
        f"{total_time / (len(target_amplitudes) * number_of_trials):.2f} s"
    )


if __name__ == "__main__":
    main()

