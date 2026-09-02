import numpy as np
import matplotlib.pyplot as plt

from src.passive_radar.geometry import (
    bistatic_range,
    bistatic_delay,
    bistatic_doppler,
)

from src.passive_radar.signal import (
    generate_reference_signal,
    generate_noise,
)

from src.passive_radar.propagation import (
    fractional_delay,
    apply_doppler,
)

from src.passive_radar.cancellation import (
    estimate_direct_path,
    cancel_direct_path,
)

from src.passive_radar.processing import (
    cross_correlate,
)


# ============================================================
# Parameters
# ============================================================

C = 299_792_458.0

sample_rate = 1e6

block_size = 2048
num_blocks = 128

N = block_size * num_blocks

carrier_frequency = 100e6

target_amplitude = 0.1
noise_amplitude = 0.5
direct_path_amplitude = 10.0


# ============================================================
# Geometry
# ============================================================

tx = np.array([0.0, 0.0])

rx = np.array([10_000.0, 0.0])

target = np.array([5_000.0, 20_000.0])

target_velocity = np.array([0.0, 250.0])


# ============================================================
# Calculate true target parameters
# ============================================================

true_direct_range = np.linalg.norm(rx - tx)

true_direct_delay = true_direct_range / C * sample_rate

true_target_range = bistatic_range(
    tx,
    target,
    rx,
)

true_target_delay = (
    bistatic_delay(
        tx,
        target,
        rx,
    )
    * sample_rate
)

true_doppler = bistatic_doppler(
    tx,
    target,
    rx,
    target_velocity,
    carrier_frequency,
)


print("=" * 60)
print("PASSIVE RADAR SIMULATION")
print("=" * 60)

print("\nTrue parameters")
print("----------------")

print(
    f"Direct-path delay: {true_direct_delay:.3f} samples"
)

print(
    f"Target delay:      {true_target_delay:.3f} samples"
)

print(
    f"Target Doppler:    {true_doppler:.2f} Hz"
)


# ============================================================
# Generate reference signal
# ============================================================

reference = generate_reference_signal(
    N,
    seed=42,
)


# ============================================================
# Generate direct path
# ============================================================

direct_path = fractional_delay(
    reference,
    true_direct_delay,
)

direct_path *= direct_path_amplitude


# ============================================================
# Generate target echo
# ============================================================

target_echo = fractional_delay(
    reference,
    true_target_delay,
)

target_echo = apply_doppler(
    target_echo,
    true_doppler,
    sample_rate,
)

target_echo *= target_amplitude


# ============================================================
# Surveillance signal
# ============================================================

noise = generate_noise(
    N,
    amplitude=noise_amplitude,
    seed=123,
)

surveillance = (
    direct_path
    + target_echo
    + noise
)


# ============================================================
# Estimate direct path
# ============================================================

integer_direct_delay = int(
    round(true_direct_delay)
)

direct_coefficient = estimate_direct_path(
    reference,
    surveillance,
    integer_direct_delay,
)


print("\nDirect-path estimation")
print("----------------------")

print(
    f"True delay:       {true_direct_delay:.3f} samples"
)

print(
    f"Integer estimate: {integer_direct_delay} samples"
)

print(
    f"Estimated amplitude: {abs(direct_coefficient):.6f}"
)

print(
    f"Estimated phase:     {np.angle(direct_coefficient):.6f} rad"
)


# ============================================================
# Direct-path cancellation
# ============================================================

cancelled = cancel_direct_path(
    reference,
    surveillance,
    integer_direct_delay,
    direct_coefficient,
)


print("\nSignal power")
print("------------")

print(
    f"Before cancellation: "
    f"{np.mean(np.abs(surveillance) ** 2):.6f}"
)

print(
    f"After cancellation:  "
    f"{np.mean(np.abs(cancelled) ** 2):.6f}"
)


# ============================================================
# Block processing
# ============================================================

blocks = cancelled.reshape(
    num_blocks,
    block_size,
)

reference_blocks = reference.reshape(
    num_blocks,
    block_size,
)


# ============================================================
# Range-Doppler processing
# ============================================================

range_doppler = np.zeros(
    (
        num_blocks,
        2 * block_size - 1,
    ),
    dtype=complex,
)


for block_index in range(num_blocks):

    correlation, lags = cross_correlate(
        reference_blocks[block_index],
        blocks[block_index],
    )

    range_doppler[block_index, :] = correlation


# ============================================================
# Doppler FFT
# ============================================================

doppler_map = np.fft.fftshift(
    np.fft.fft(
        range_doppler,
        axis=0,
    ),
    axes=0,
)


doppler_power = (
    20
    * np.log10(
        np.abs(doppler_map) + 1e-12
    )
)


# ============================================================
# Find detection
# ============================================================

peak_index = np.unravel_index(
    np.argmax(doppler_power),
    doppler_power.shape,
)

doppler_index = peak_index[0]
delay_index = peak_index[1]

detected_delay = lags[delay_index]

doppler_frequency_axis = np.fft.fftshift(
    np.fft.fftfreq(
        num_blocks,
        d=block_size / sample_rate,
    )
)

detected_doppler = (
    doppler_frequency_axis[doppler_index]
)


print("\nDetection")
print("---------")

print(
    f"True delay:      {true_target_delay:.3f} samples"
)

print(
    f"Detected delay:  {detected_delay} samples"
)

print(
    f"True Doppler:    {true_doppler:.2f} Hz"
)

print(
    f"Detected Doppler: {detected_doppler:.2f} Hz"
)


# ============================================================
# Plot range-Doppler map
# ============================================================

plt.figure(figsize=(10, 6))

plt.imshow(
    doppler_power,
    aspect="auto",
    origin="lower",
    extent=[
        lags[0],
        lags[-1],
        doppler_frequency_axis[0],
        doppler_frequency_axis[-1],
    ],
)

plt.xlabel("Delay [samples]")
plt.ylabel("Doppler [Hz]")
plt.title("Passive Radar Range-Doppler Map")

plt.colorbar(
    label="Magnitude [dB]"
)

plt.tight_layout()

plt.show()