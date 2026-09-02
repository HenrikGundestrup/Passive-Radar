import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Functions
# ============================================================

def fractional_delay(signal, delay):

    N = len(signal)

    pad = int(np.ceil(delay)) + 10

    padded = np.pad(
        signal,
        (pad, pad),
        mode="constant"
    )

    M = len(padded)

    spectrum = np.fft.fft(padded)

    frequencies = np.fft.fftfreq(M)

    phase_shift = np.exp(
        -1j * 2 * np.pi * frequencies * delay
    )

    delayed = np.fft.ifft(
        spectrum * phase_shift
    )

    delayed = delayed[
        pad:pad + N
    ]

    return delayed

'''
def fractional_delay(signal, delay):

    N = len(signal)

    n = np.arange(N)

    delayed_n = n - delay

    delayed_signal = np.interp(
        delayed_n,
        n,
        signal.real,
        left=0,
        right=0
    ) + 1j * np.interp(
        delayed_n,
        n,
        signal.imag,
        left=0,
        right=0
    )

    return delayed_signal
'''

# ============================================================
# PARAMETERS
# ============================================================

sample_rate = 1e6

block_size = 2048
num_blocks = 128

N = block_size * num_blocks

target_amplitude = 0.02
noise_amplitude = 0.5
direct_path_amplitude = 10.0


# ============================================================
# BISTATIC RADAR GEOMETRY
# ============================================================

c = 299_792_458
carrier_frequency = 100e6
wavelength = c / carrier_frequency

tx_position = np.array([0.0, 0.0])
rx_position = np.array([10_000.0, 0.0])

target_position = np.array([5_000.0, 20_000.0])
target_velocity = np.array([0.0, 250.0])


# ============================================================
# BISTATIC RANGE
# ============================================================

tx_vector = target_position - tx_position
rx_vector = target_position - rx_position

tx_range = np.linalg.norm(tx_vector)
rx_range = np.linalg.norm(rx_vector)

tx_unit = tx_vector / tx_range
rx_unit = rx_vector / rx_range

bistatic_range = tx_range + rx_range


# ============================================================
# DIRECT PATH
# ============================================================

direct_path_range = np.linalg.norm(
    rx_position - tx_position
)

direct_path_time = (
    direct_path_range / c
)

direct_path_delay = (
    direct_path_time * sample_rate
)


# ============================================================
# TARGET PROPAGATION DELAY
# ============================================================

propagation_time = (
    bistatic_range / c
)

delay_samples = (
    propagation_time * sample_rate
)


# ============================================================
# DOPPLER SHIFT
# ============================================================

doppler = (
    np.dot(
        target_velocity,
        tx_unit + rx_unit
    )
    / wavelength
)


# ============================================================
# PRINT GEOMETRY
# ============================================================

print()
print("=" * 60)
print("BISTATIC RADAR GEOMETRY")
print("=" * 60)

print()

print(
    f"TX → Target range:   "
    f"{tx_range:.2f} m"
)

print(
    f"Target → RX range:   "
    f"{rx_range:.2f} m"
)

print(
    f"Bistatic range:      "
    f"{bistatic_range:.2f} m"
)

print()

print(
    f"Direct path range:   "
    f"{direct_path_range:.2f} m"
)

print(
    f"Direct path delay:   "
    f"{direct_path_delay:.3f} samples"
)

print()

print(
    f"Propagation time:    "
    f"{propagation_time * 1e6:.3f} µs"
)

print(
    f"Target delay:        "
    f"{delay_samples:.3f} samples"
)

print()

print(
    f"Target velocity:     "
    f"{np.linalg.norm(target_velocity):.2f} m/s"
)

print(
    f"Carrier frequency:   "
    f"{carrier_frequency:.2f} Hz"
)

print(
    f"Wavelength:          "
    f"{wavelength:.3f} m"
)

print(
    f"Doppler shift:       "
    f"{doppler:.2f} Hz"
)


# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(42)


# ============================================================
# REFERENCE SIGNAL
# ============================================================

reference = (
    rng.normal(size=N)
    + 1j * rng.normal(size=N)
) / np.sqrt(2)


# ============================================================
# TARGET
# ============================================================

target = fractional_delay(
    reference,
    delay_samples
)

target *= target_amplitude


# ============================================================
# DOPPLER
# ============================================================

n = np.arange(N)

doppler_rotation = np.exp(
    1j * 2 * np.pi * doppler * n / sample_rate
)

target *= doppler_rotation


# ============================================================
# DIRECT PATH
# ============================================================

direct_path = fractional_delay(
    reference,
    direct_path_delay
)

direct_path *= direct_path_amplitude


# ============================================================
# RECEIVER NOISE
# ============================================================

noise = (
    rng.normal(size=N)
    + 1j * rng.normal(size=N)
) / np.sqrt(2)

noise *= noise_amplitude


# ============================================================
# SURVEILLANCE SIGNAL
# ============================================================

surveillance = (
    target
    + direct_path
    + noise
)


# ============================================================
# AUTOMATIC DIRECT-PATH DELAY ESTIMATION
# ============================================================

# We use the geometry only to define a search region.
# The actual delay is estimated from the received signal.

search_width = 20  # samples


search_start = int(
    np.floor(
        direct_path_delay - search_width
    )
)

search_end = int(
    np.ceil(
        direct_path_delay + search_width
    )
)

search_start = max(
    0,
    search_start
)

search_end = min(
    N - 1,
    search_end
)


# ============================================================
# CROSS-CORRELATION
# ============================================================

correlation = np.correlate(
    surveillance,
    reference,
    mode="full"
)


# ============================================================
# CORRELATION LAG AXIS
# ============================================================

correlation_lags = np.arange(
    -(N - 1),
    N
)


# ============================================================
# POSITIVE DELAYS
# ============================================================

positive_delay_mask = (
    correlation_lags >= 0
)

positive_lags = (
    correlation_lags[
        positive_delay_mask
    ]
)

positive_correlation = (
    correlation[
        positive_delay_mask
    ]
)


# ============================================================
# DIRECT-PATH SEARCH WINDOW
# ============================================================

direct_search_mask = (
    (positive_lags >= search_start)
    &
    (positive_lags <= search_end)
)

search_correlation = np.abs(
    positive_correlation[
        direct_search_mask
    ]
)

search_lags = (
    positive_lags[
        direct_search_mask
    ]
)


# ============================================================
# FIND INTEGER DIRECT-PATH DELAY
# ============================================================

direct_peak_index = np.argmax(
    search_correlation
)

integer_delay = (
    search_lags[
        direct_peak_index
    ]
)


# ============================================================
# FINE DIRECT-PATH DELAY SEARCH
# ============================================================

# Search ±0.5 samples around the integer peak.

fine_step = 0.001

fine_delays = np.arange(
    integer_delay - 0.5,
    integer_delay + 0.5 + fine_step,
    fine_step
)


errors = []
alphas = []


for test_delay in fine_delays:

    test_reference = fractional_delay(
        reference,
        test_delay
    )

    # --------------------------------------------------------
    # Least-squares complex amplitude
    # --------------------------------------------------------

    test_alpha = (
        np.vdot(
            test_reference,
            surveillance
        )
        /
        np.vdot(
            test_reference,
            test_reference
        )
    )

    # --------------------------------------------------------
    # Reconstruct direct path
    # --------------------------------------------------------

    reconstructed = (
        test_alpha
        * test_reference
    )

    # --------------------------------------------------------
    # Residual error
    # --------------------------------------------------------

    residual = (
        surveillance
        - reconstructed
    )

    error = np.vdot(
        residual,
        residual
    ).real

    errors.append(
        error
    )

    alphas.append(
        test_alpha
    )


# ============================================================
# SELECT BEST DELAY
# ============================================================

best_index = np.argmin(
    errors
)

estimated_direct_delay = (
    fine_delays[
        best_index
    ]
)

alpha = (
    alphas[
        best_index
    ]
)


# ============================================================
# RECONSTRUCT ESTIMATED DIRECT PATH
# ============================================================

direct_reference = fractional_delay(
    reference,
    estimated_direct_delay
)

estimated_direct_path = (
    alpha
    * direct_reference
)


# ============================================================
# DIRECT-PATH CANCELLATION
# ============================================================

surveillance_clean = (
    surveillance
    - estimated_direct_path
)


# ============================================================
# DIRECT-PATH DIAGNOSTICS
# ============================================================

print()
print("=" * 60)
print("AUTOMATIC DIRECT-PATH ESTIMATION")
print("=" * 60)

print()

print(
    f"True direct delay:       "
    f"{direct_path_delay:.3f} samples"
)

print(
    f"Integer delay estimate:  "
    f"{integer_delay:.0f} samples"
)

print(
    f"Fine delay estimate:     "
    f"{estimated_direct_delay:.3f} samples"
)

print()

print(
    f"True direct amplitude:   "
    f"{direct_path_amplitude:.6f}"
)

print(
    f"Estimated amplitude:     "
    f"{np.abs(alpha):.6f}"
)

print(
    f"Estimated phase:         "
    f"{np.angle(alpha):.6f} rad"
)


# ============================================================
# FFT SIZE FOR LINEAR CORRELATION
# ============================================================

fft_size = (
    2 * block_size - 1
)


# ============================================================
# STORAGE
# ============================================================

delay_profiles = []


# ============================================================
# PROCESS BLOCKS
# ============================================================

for block in range(num_blocks):

    start = block * block_size
    end = start + block_size

    ref_block = reference[start:end]

    # Use direct-path-cancelled surveillance signal.

    surv_block = (
        surveillance_clean[start:end]
    )


    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    ref_fft = np.fft.fft(
        ref_block,
        fft_size
    )

    surv_fft = np.fft.fft(
        surv_block,
        fft_size
    )


    # --------------------------------------------------------
    # CROSS-CORRELATION
    # --------------------------------------------------------

    correlation_fft = (
        surv_fft
        * np.conj(ref_fft)
    )

    correlation = np.fft.ifft(
        correlation_fft
    )


    # --------------------------------------------------------
    # REORDER CORRELATION
    # --------------------------------------------------------

    correlation = np.concatenate(
        (
            correlation[-(block_size - 1):],
            correlation[:block_size]
        )
    )


    # --------------------------------------------------------
    # KEEP COMPLEX CORRELATION
    # --------------------------------------------------------

    delay_profiles.append(
        correlation
    )


# ============================================================
# CONVERT TO ARRAY
# ============================================================

delay_profiles = np.array(
    delay_profiles
)


# ============================================================
# SLOW-TIME DOPPLER FFT
# ============================================================

doppler_map = np.fft.fft(
    delay_profiles,
    axis=0
)

doppler_map = np.fft.fftshift(
    doppler_map,
    axes=0
)


# ============================================================
# DOPPLER AXIS
# ============================================================

block_time = (
    block_size / sample_rate
)

doppler_axis = np.fft.fftshift(
    np.fft.fftfreq(
        num_blocks,
        d=block_time
    )
)


# ============================================================
# MAGNITUDE
# ============================================================

doppler_map_magnitude = np.abs(
    doppler_map
)


# ============================================================
# DELAY AXIS
# ============================================================

delay_axis = np.arange(
    -(block_size - 1),
    block_size
)


# ============================================================
# FIND PEAK
# ============================================================

peak_index = np.unravel_index(
    np.argmax(
        doppler_map_magnitude
    ),
    doppler_map_magnitude.shape
)

doppler_index = peak_index[0]

delay_index = peak_index[1]


detected_doppler = (
    doppler_axis[
        doppler_index
    ]
)

detected_delay = (
    delay_axis[
        delay_index
    ]
)


# ============================================================
# PRINT FINAL RESULT
# ============================================================

print()
print("=" * 60)
print("PASSIVE RADAR RANGE-DOPPLER")
print("=" * 60)

print()

print(
    f"True delay:       "
    f"{delay_samples:.3f} samples"
)

print(
    f"Detected delay:   "
    f"{detected_delay} samples"
)

print()

print(
    f"True Doppler:     "
    f"{doppler:.2f} Hz"
)

print(
    f"Detected Doppler: "
    f"{detected_doppler:.2f} Hz"
)

print()

print(
    f"Target amplitude: "
    f"{target_amplitude}"
)

print(
    f"Direct path amplitude: "
    f"{direct_path_amplitude}"
)

print(
    f"Noise amplitude:  "
    f"{noise_amplitude}"
)


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.imshow(
    20 * np.log10(
        doppler_map_magnitude
        / np.max(
            doppler_map_magnitude
        )
        + 1e-12
    ),

    aspect="auto",

    extent=[
        delay_axis[0],
        delay_axis[-1],
        doppler_axis[0],
        doppler_axis[-1]
    ],

    origin="lower"
)

plt.colorbar(
    label="Relative magnitude [dB]"
)

plt.axvline(
    delay_samples,
    linestyle="--",
    label="True delay"
)

plt.axhline(
    doppler,
    linestyle="--",
    label="True Doppler"
)

plt.xlabel(
    "Delay [samples]"
)

plt.ylabel(
    "Doppler [Hz]"
)

plt.title(
    "Passive Radar Range-Doppler Map"
)

plt.legend()

plt.tight_layout()

plt.show()