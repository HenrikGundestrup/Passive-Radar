import numpy as np
import matplotlib.pyplot as plt

from geometry import (
    bistatic_range,
    bistatic_delay,
    bistatic_doppler,
)

from radar_signal import (
    generate_reference_signal,
    generate_noise,
)

from propagation import (
    fractional_delay,
    apply_doppler,
)

from cancellation import (
    estimate_direct_path,
    cancel_direct_path,
)

from processing import (
    range_doppler_processing,
    detect_peak,
)


# ============================================================
# Parameters
# ============================================================

sample_rate = 1e6
block_size = 2048
num_blocks = 128

N = block_size * num_blocks

carrier_frequency = 100e6

C = 299_792_458.0

target_amplitude = 0.02
direct_path_amplitude = 10.0
noise_amplitude = 0.5


# ============================================================
# Geometry
# ============================================================

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
    rx_position
)

true_target_delay = bistatic_delay(
    tx_position,
    target_position,
    rx_position
)

true_target_delay_samples = (
    true_target_delay * sample_rate
)

direct_path_range = np.linalg.norm(
    rx_position - tx_position
)

direct_path_delay = (
    direct_path_range / C * sample_rate
)

doppler = bistatic_doppler(
    tx_position,
    target_position,
    rx_position,
    target_velocity,
    carrier_frequency
)

wavelength = C / carrier_frequency


# ============================================================
# Print geometry
# ============================================================

print()
print("=" * 60)
print("BISTATIC RADAR GEOMETRY")
print("=" * 60)

print(f"TX → Target range:   {tx_range:.2f} m")
print(f"Target → RX range:   {rx_range:.2f} m")
print(f"Bistatic range:      {true_target_range:.2f} m")

print()
print(f"Direct path range:   {direct_path_range:.2f} m")
print(f"Direct path delay:   {direct_path_delay:.3f} samples")

print()
print(f"Propagation time:    {true_target_delay * 1e6:.3f} µs")
print(f"Target delay:        {true_target_delay_samples:.3f} samples")

print()
print(f"Target velocity:     {np.linalg.norm(target_velocity):.2f} m/s")
print(f"Carrier frequency:   {carrier_frequency:.2f} Hz")
print(f"Wavelength:          {wavelength:.3f} m")
print(f"Doppler shift:       {doppler:.2f} Hz")


# ============================================================
# Generate reference signal
# ============================================================

reference = generate_reference_signal(
    N,
    seed=42
)


# ============================================================
# Generate target signal
# ============================================================

target = fractional_delay(
    reference,
    true_target_delay_samples
)

target *= target_amplitude

target = apply_doppler(
    target,
    doppler,
    sample_rate
)


# ============================================================
# Generate direct path
# ============================================================

direct_path = fractional_delay(
    reference,
    direct_path_delay
)

direct_path *= direct_path_amplitude


# ============================================================
# Generate noise
# ============================================================

noise = generate_noise(
    N,
    amplitude=noise_amplitude,
    seed=123
)


# ============================================================
# Surveillance signal
# ============================================================

surveillance = (
    target
    + direct_path
    + noise
)


# ============================================================
# Automatic direct-path estimation
# ============================================================

estimated_direct_delay, direct_coefficient = (
    estimate_direct_path(
        reference,
        surveillance,
        direct_path_delay
    )
)


print()
print("=" * 60)
print("AUTOMATIC DIRECT-PATH ESTIMATION")
print("=" * 60)

print(
    f"True direct delay:       "
    f"{direct_path_delay:.3f} samples"
)

print(
    f"Estimated direct delay:  "
    f"{estimated_direct_delay:.3f} samples"
)

print(
    f"True direct amplitude:   "
    f"{direct_path_amplitude:.6f}"
)

print(
    f"Estimated amplitude:     "
    f"{abs(direct_coefficient):.6f}"
)

print(
    f"Estimated phase:         "
    f"{np.angle(direct_coefficient):.6f} rad"
)


# ============================================================
# Direct-path cancellation
# ============================================================

surveillance_cancelled = cancel_direct_path(
    reference,
    surveillance,
    estimated_direct_delay,
    direct_coefficient
)


# ============================================================
# Range-Doppler processing
# ============================================================

doppler_map, delay_axis, doppler_axis = (
    range_doppler_processing(
        reference,
        surveillance_cancelled,
        sample_rate,
        block_size,
        num_blocks,
    )
)


# ============================================================
# Target detection
# ============================================================

detected_delay, detected_doppler = detect_peak(
    doppler_map,
    delay_axis,
    doppler_axis,
)


print()
print("=" * 60)
print("PASSIVE RADAR RANGE-DOPPLER")
print("=" * 60)

print(
    f"True delay:       "
    f"{true_target_delay_samples:.3f} samples"
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
print(f"Target amplitude:      {target_amplitude}")
print(f"Direct path amplitude: {direct_path_amplitude}")
print(f"Noise amplitude:       {noise_amplitude}")


# ============================================================
# Plot range-Doppler map
# ============================================================

power_db = 20 * np.log10(
    np.abs(doppler_map)
    / np.max(np.abs(doppler_map))
    + 1e-12
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
        doppler_axis[-1]
    ]
)

plt.colorbar(
    label="Relative power (dB)"
)

plt.xlabel("Delay (samples)")
plt.ylabel("Doppler (Hz)")

plt.title(
    "Passive Radar Range-Doppler Map"
)

plt.show()

