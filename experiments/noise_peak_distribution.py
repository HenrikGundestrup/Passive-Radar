import time
import numpy as np

from simulations.basic_simulation import run_simulation


def main():

    start_time = time.perf_counter()

    number_of_trials = 100

    peak_powers = []

    print()
    print("=" * 70)
    print("NOISE-ONLY PEAK POWER DISTRIBUTION")
    print("=" * 70)

    print(f"Trials: {number_of_trials}")
    print()

    # ========================================================
    # Noise-only experiment
    # ========================================================

    for trial in range(number_of_trials):

        result = run_simulation(
            show_plot=False,
            noise_seed=trial,
            target_amplitude=0.0,
        )

        peak_powers.append(
            result["peak_power"]
        )

    peak_powers = np.array(
        peak_powers
    )

    # ========================================================
    # Statistics
    # ========================================================

    minimum = np.min(
        peak_powers
    )

    maximum = np.max(
        peak_powers
    )

    mean = np.mean(
        peak_powers
    )

    median = np.median(
        peak_powers
    )

    percentile_95 = np.percentile(
        peak_powers,
        95
    )

    percentile_99 = np.percentile(
        peak_powers,
        99
    )

    total_time = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # Results
    # ========================================================

    print("=" * 70)
    print("PEAK POWER STATISTICS")
    print("=" * 70)

    print(
        f"Minimum:          {minimum:.6e}"
    )

    print(
        f"Mean:             {mean:.6e}"
    )

    print(
        f"Median:           {median:.6e}"
    )

    print(
        f"95th percentile:  {percentile_95:.6e}"
    )

    print(
        f"99th percentile:  {percentile_99:.6e}"
    )

    print(
        f"Maximum:          {maximum:.6e}"
    )

    print("-" * 70)

    print(
        f"Total experiment time: "
        f"{total_time:.2f} s"
    )

    print(
        f"Average time per trial: "
        f"{total_time / number_of_trials:.2f} s"
    )


if __name__ == "__main__":
    main()