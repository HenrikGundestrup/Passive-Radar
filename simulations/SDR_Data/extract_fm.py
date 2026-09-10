import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt


# ============================================================
# Settings
# ============================================================

CENTER_FREQUENCY = 100e6
SAMPLE_RATE = 2.4e6

TARGET_FREQUENCY = 99.9e6

INPUT_FILE = "recording.npy"
OUTPUT_FILE = "fm_reference.npy"


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

print(f"Frequency offset: {frequency_offset / 1e3:.1f} kHz")

n = np.arange(len(samples))

# Shift -100 kHz signal to 0 Hz
shifted = samples * np.exp(
    -1j * 2 * np.pi * frequency_offset * n / SAMPLE_RATE
)


# ============================================================
# Low-pass filter
# ============================================================

# FM broadcast bandwidth is roughly 200 kHz.
# Keep approximately +/- 100 kHz.

cutoff = 100e3

sos = butter(
    6,
    cutoff,
    btype="lowpass",
    fs=SAMPLE_RATE,
    output="sos"
)

filtered = sosfilt(sos, shifted)


# ============================================================
# Save
# ============================================================

filtered = filtered.astype(np.complex64)

np.save(OUTPUT_FILE, filtered)

print()
print(f"Saved: {OUTPUT_FILE}")
print(f"Samples: {len(filtered):,}")
print(f"Size: {filtered.nbytes / 1024**2:.1f} MB")


# ============================================================
# Plot extracted signal
# ============================================================

NFFT = 2**18

data = filtered[:NFFT]

window = np.hanning(len(data))

spectrum = np.fft.fftshift(
    np.fft.fft(data * window)
)

frequencies = np.fft.fftshift(
    np.fft.fftfreq(len(data), 1 / SAMPLE_RATE)
)

power_db = 20 * np.log10(
    np.abs(spectrum) + 1e-12
)


plt.figure(figsize=(10, 5))

plt.plot(
    frequencies / 1e3,
    power_db
)

plt.xlim(-200, 200)

plt.xlabel("Frequency relative to 99.9 MHz [kHz]")
plt.ylabel("Magnitude [dB]")
plt.title("Extracted 99.9 MHz FM signal")

plt.grid()

plt.tight_layout()
plt.show()