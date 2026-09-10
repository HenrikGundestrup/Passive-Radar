import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Settings
# ============================================================

CENTER_FREQUENCY = 100e6
TARGET_FREQUENCY = 99.9e6
SAMPLE_RATE = 2.4e6

INPUT_FILE = "recording.npy"

# Processing
BLOCK_SIZE = 1024
NUM_BLOCKS = 128

# Only show positive delays
MAX_DELAY = 2000

# Frequency range shown in the map
MAX_DOPPLER = 1000


# ============================================================
# Load recording
# ============================================================

print("Loading recording...")

samples = np.load(INPUT_FILE)

print(f"Samples:  {len(samples):,}")
print(f"Duration: {len(samples) / SAMPLE_RATE:.2f} s")


# ============================================================
# Shift 99.9 MHz to baseband
# ============================================================

frequency_offset = TARGET_FREQUENCY - CENTER_FREQUENCY

n = np.arange(len(samples))

samples = samples * np.exp(
    -1j * 2 * np.pi * frequency_offset * n / SAMPLE_RATE
)


# ============================================================
# Remove DC component
# ============================================================

samples = samples - np.mean(samples)


# ============================================================
# Use a section of the recording
# ============================================================

N = BLOCK_SIZE * NUM_BLOCKS

if len(samples) < N:
    raise ValueError("Recording is too short.")

samples = samples[:N]


# ============================================================
# Build delay-Doppler map
# ============================================================

print("Calculating delay-Doppler map...")

delay_profiles = []

fft_size = 2 * BLOCK_SIZE - 1

for block in range(NUM_BLOCKS):

    start = block * BLOCK_SIZE
    end = start + BLOCK_SIZE

    reference = samples[start:end]
    surveillance = samples[start:end]

    reference_fft = np.fft.fft(
        reference,
        fft_size
    )

    surveillance_fft = np.fft.fft(
        surveillance,
        fft_size
    )

    correlation = np.fft.ifft(
        surveillance_fft
        * np.conj(reference_fft)
    )

    # Rearrange correlation so that:
    # negative delays | zero | positive delays

    correlation = np.concatenate(
        (
            correlation[-(BLOCK_SIZE - 1):],
            correlation[:BLOCK_SIZE]
        )
    )

    delay_profiles.append(correlation)


delay_profiles = np.asarray(delay_profiles)


# ============================================================
# Doppler processing
# ============================================================

doppler_map = np.fft.fftshift(
    np.fft.fft(
        delay_profiles,
        axis=0
    ),
    axes=0
)


# ============================================================
# Axes
# ============================================================

delay_axis = np.arange(
    -(BLOCK_SIZE - 1),
    BLOCK_SIZE
)

slow_time_sample_rate = SAMPLE_RATE / BLOCK_SIZE

doppler_axis = np.fft.fftshift(
    np.fft.fftfreq(
        NUM_BLOCKS,
        d=1 / slow_time_sample_rate
    )
)


# ============================================================
# Select positive delays
# ============================================================

delay_mask = (
    (delay_axis >= 0)
    & (delay_axis <= MAX_DELAY)
)

doppler_mask = (
    np.abs(doppler_axis) <= MAX_DOPPLER
)

map_display = np.abs(
    doppler_map[
        doppler_mask
    ][:, delay_mask]
)

map_db = 20 * np.log10(
    map_display + 1e-12
)

display_delay = delay_axis[delay_mask]
display_doppler = doppler_axis[doppler_mask]


# ============================================================
# Normalize to strongest peak
# ============================================================

map_db = map_db - np.max(map_db)


# ============================================================
# Plot delay-Doppler map
# ============================================================

plt.figure(figsize=(11, 7))

plt.imshow(
    map_db,
    aspect="auto",
    origin="lower",
    extent=[
        display_delay[0],
        display_delay[-1],
        display_doppler[0],
        display_doppler[-1]
    ],
    vmin=-60,
    vmax=0
)

plt.xlabel("Delay [samples]")
plt.ylabel("Doppler [Hz]")
plt.title("Self-Correlation Delay-Doppler Map")

plt.colorbar(
    label="Relative magnitude [dB]"
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.tight_layout()
plt.show()


# ============================================================
# Print strongest peaks
# ============================================================

print()
print("=" * 60)
print("STRONGEST PEAKS")
print("=" * 60)

flat_indices = np.argsort(
    map_display.ravel()
)[::-1]

found = 0

for index in flat_indices:

    doppler_index, delay_index = np.unravel_index(
        index,
        map_display.shape
    )

    delay = display_delay[delay_index]
    doppler = display_doppler[doppler_index]

    # Skip the exact direct/self-correlation peak
    if delay == 0 and doppler == 0:
        continue

    magnitude_db = map_db[
        doppler_index,
        delay_index
    ]

    print(
        f"Delay: {delay:6.0f} samples   "
        f"Doppler: {doppler:8.2f} Hz   "
        f"Level: {magnitude_db:7.2f} dB"
    )

    found += 1

    if found >= 10:
        break