import time
import numpy as np

from simulations.basic_simulation import run_simulation


def main():

    start_time = time.perf_counter()

    number_of_trials = 20

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

    print()
    print("=" * 70)
    print("PEAK POWER VS TARGET AMPLITUDE")
    print("=" * 70)

    print(f"Trials per amplitude: {number_of_trials}")
    print()

    results = []

    # ========================================================
    # Experiment
    # ========================================================

    for target_amplitude in target_amplitudes:

        peak_powers = []
        snr_values = []

        for trial in range(number_of_trials):

            result = run_simulation(
                show_plot=False,
                noise_seed=trial,
                target_amplitude=target_amplitude,
            )

            peak_powers.append(
                result["peak_power"]
            )

            snr_values.append(
                result["snr_db"]
            )

        peak_powers = np.array(
            peak_powers
        )

        snr_values = np.array(
            snr_values
        )

        mean_peak_power = np.mean(
            peak_powers
        )

        median_peak_power = np.median(
            peak_powers
        )

        minimum_peak_power = np.min(
            peak_powers
        )

        maximum_peak_power = np.max(
            peak_powers
        )

        mean_snr = np.mean(
            snr_values
        )

        results.append(
            {
                "target_amplitude": target_amplitude,
                "mean_peak_power": mean_peak_power,
                "median_peak_power": median_peak_power,
                "minimum_peak_power": minimum_peak_power,
                "maximum_peak_power": maximum_peak_power,
                "mean_snr": mean_snr,
            }
        )

        print(
            f"Amplitude {target_amplitude:6.3f} | "
            f"SNR {mean_snr:7.2f} dB | "
            f"Peak power "
            f"{mean_peak_power:.6e} | "
            f"Range "
            f"[{minimum_peak_power:.6e}, "
            f"{maximum_peak_power:.6e}]"
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
        f"{'Mean peak':>18} "
        f"{'Median peak':>18}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['target_amplitude']:12.4f} "
            f"{result['mean_snr']:12.2f} "
            f"{result['mean_peak_power']:18.6e} "
            f"{result['median_peak_power']:18.6e}"
        )

    print("-" * 70)

    total_time = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Total experiment time: "
        f"{total_time:.2f} s"
    )

    print(
        f"Average time per trial: "
        f"{total_time / 
        (number_of_trials * len(target_amplitudes)):.2f} s"
    )


if __name__ == "__main__":
    main()