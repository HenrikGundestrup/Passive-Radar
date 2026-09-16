import time
import numpy as np

from simulations.basic_simulation import run_simulation


def main():

    start_time = time.perf_counter()

    number_of_trials = 100

    threshold = 8.8e5

    target_amplitudes = [
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.010,
    ]

    results = []

    print()
    print("=" * 70)
    print("DETECTION PROBABILITY AT FIXED THRESHOLD")
    print("=" * 70)

    print(
        f"Trials per amplitude: {number_of_trials}"
    )

    print(
        f"Detection threshold:  {threshold:.2e}"
    )

    print()

    # ========================================================
    # Detection experiment
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

            peak_power = result["peak_power"]

            snr_values.append(
                result["snr_db"]
            )

            if peak_power >= threshold:
                detections += 1

        probability = (
            detections
            / number_of_trials
        )

        mean_snr = np.mean(
            snr_values
        )

        results.append(
            {
                "target_amplitude": target_amplitude,
                "snr_db": mean_snr,
                "detections": detections,
                "probability": probability,
            }
        )

        print(
            f"Amplitude {target_amplitude:6.3f} | "
            f"SNR {mean_snr:7.2f} dB | "
            f"Detections "
            f"{detections:3d}/"
            f"{number_of_trials} | "
            f"P_D "
            f"{probability * 100:6.1f} %"
        )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"{'Amplitude':>12} "
        f"{'SNR [dB]':>12} "
        f"{'Detections':>14} "
        f"{'P_D':>12}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['target_amplitude']:12.4f} "
            f"{result['snr_db']:12.2f} "
            f"{result['detections']:8d}/"
            f"{number_of_trials:<5d} "
            f"{result['probability'] * 100:10.1f} %"
        )

    print("-" * 70)

    total_time = (
        time.perf_counter()
        - start_time
    )

    total_trials = (
        number_of_trials
        * len(target_amplitudes)
    )

    print(
        f"Total experiment time: "
        f"{total_time:.2f} s"
    )

    print(
        f"Average time per trial: "
        f"{total_time / total_trials:.2f} s"
    )


if __name__ == "__main__":
    main()