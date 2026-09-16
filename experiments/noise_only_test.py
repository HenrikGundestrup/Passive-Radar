from simulations.basic_simulation import run_simulation


def main():

    number_of_trials = 10

    print()
    print("=" * 70)
    print("NOISE-ONLY TEST")
    print("=" * 70)

    print(f"Trials: {number_of_trials}")
    print("Target amplitude: 0.0")
    print()

    for trial in range(number_of_trials):

        result = run_simulation(
            show_plot=False,
            noise_seed=trial,
            target_amplitude=0.0,
        )

        print(
            f"Trial {trial + 1:2d} | "
            f"Peak power: {result['peak_power']:.6e} | "
            f"Delay: {result['detected_delay']:4d} | "
            f"Doppler: {result['detected_doppler']:8.2f} Hz"
        )


if __name__ == "__main__":
    main()