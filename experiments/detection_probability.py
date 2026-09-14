from simulations.basic_simulation import run_simulation


def main():

    number_of_trials = 10

    print()
    print("=" * 60)
    print("MONTE CARLO DETECTION EXPERIMENT")
    print("=" * 60)

    for trial in range(number_of_trials):

        result = run_simulation(
            show_plot=False,
            noise_seed=trial,
        )

        print(
            f"Trial {trial + 1:2d}: "
            f"delay = {result['detected_delay']:4d}, "
            f"Doppler = {result['detected_doppler']:8.2f} Hz"
        )


if __name__ == "__main__":
    main()