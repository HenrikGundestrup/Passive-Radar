import time
import numpy as np

from simulations.basic_simulation import run_simulation


def main():

    start_time = time.perf_counter()

    number_of_trials = 1000

    thresholds = np.array([
        5.5e5,
        6.0e5,
        6.5e5,
        7.0e5,
        7.5e5,
        8.0e5,
        8.5e5,
        9.0e5,
        1.0e6,
    ])

    false_alarms = np.zeros(
        len(thresholds),
        dtype=int,
    )

    print()
    print("=" * 70)
    print("FALSE ALARM PROBABILITY EXPERIMENT")
    print("=" * 70)

    print(f"Noise-only trials: {number_of_trials}")
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

        peak_power = result["peak_power"]

        false_alarms += (
            peak_power >= thresholds
        )

        # Progress every 100 trials

        if (
            (trial + 1) % 100 == 0
        ):

            print(
                f"Completed "
                f"{trial + 1:4d}/"
                f"{number_of_trials}"
            )

    # ========================================================
    # Calculate P_FA
    # ========================================================

    p_fa = (
        false_alarms
        / number_of_trials
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 70)
    print("FALSE ALARM PROBABILITY")
    print("=" * 70)

    print(
        f"{'Threshold':>15} "
        f"{'False alarms':>15} "
        f"{'P_FA':>12}"
    )

    print("-" * 70)

    for threshold, alarms, probability in zip(
        thresholds,
        false_alarms,
        p_fa,
    ):

        print(
            f"{threshold:15.6e} "
            f"{alarms:15d} "
            f"{probability:11.4f}"
        )

    print("-" * 70)

    # ========================================================
    # Total time
    # ========================================================

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
        f"{total_time / number_of_trials:.2f} s"
    )


if __name__ == "__main__":
    main()