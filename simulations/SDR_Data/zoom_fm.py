import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Settings
# ============================================================

CENTER_FREQUENCY = 100e6       # SDR center frequency
SAMPLE_RATE = 2.4e6            # 2.4 MS/s

TARGET_FREQUENCY = 99.9e6      # Signal we want to inspect
SPAN = 300e3                   # +/- 300 kHz

INPUT_FILE = "recording.npy"


# ============================================================
# Load recording
# ============================================================

samples = np.load(INPUT_FILE)

print(f"Loaded {len(samples):,} samples")
print(f"Duration: {len(samples) / SAMPLE_RATE:.2f} s")


# ============================================================
# Spectrum
# ============================================================

NFFT = 2**18

# Use first part of recording
data = samples[:NFFT]

window = np.hanning(len(data))
spectrum = np.fft.fftshift(
    np.fft.fft(data * window)
)

frequencies = np.fft.fftshift(
    np.fft.fftfreq(len(data), 1 / SAMPLE_RATE)
)

# Convert baseband frequency to RF frequency
rf_frequencies = CENTER_FREQUENCY + frequencies

power_db = 20 * np.log10(
    np.abs(spectrum) + 1e-12
)

# Zoom around target
mask = (
    (rf_frequencies >= TARGET_FREQUENCY - SPAN)
    & (rf_frequencies <= TARGET_FREQUENCY + SPAN)
)


# ============================================================
# Waterfall
# ============================================================

waterfall_nfft = 4096
num_rows = 300

max_samples = min(
    len(samples),
    waterfall_nfft * num_rows
)

data = samples[:max_samples]

# Make complete blocks
data = data[:len(data) // waterfall_nfft * waterfall_nfft]

blocks = data.reshape(-1, waterfall_nfft)

window = np.hanning(waterfall_nfft)

waterfall = np.fft.fftshift(
    np.fft.fft(blocks * window, axis=1),
    axes=1
)

waterfall_db = 20 * np.log10(
    np.abs(waterfall) + 1e-12
)

wf_freq = CENTER_FREQUENCY + np.fft.fftshift(
    np.fft.fftfreq(waterfall_nfft, 1 / SAMPLE_RATE)
)

mask_wf = (
    (wf_freq >= TARGET_FREQUENCY - SPAN)
    & (wf_freq <= TARGET_FREQUENCY + SPAN)
)

waterfall_db = waterfall_db[:, mask_wf]
wf_freq = wf_freq[mask_wf]


# ============================================================
# Plot spectrum
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    rf_frequencies[mask] / 1e6,
    power_db[mask]
)

plt.xlabel("Frequency [MHz]")
plt.ylabel("Magnitude [dB]")
plt.title("Spectrum around 99.9 MHz")
plt.grid()

plt.axvline(
    TARGET_FREQUENCY / 1e6,
    linestyle="--",
    label="99.9 MHz"
)

plt.legend()
plt.tight_layout()


# ============================================================
# Plot waterfall
# ============================================================

plt.figure(figsize=(10, 6))

plt.imshow(
    waterfall_db,
    aspect="auto",
    origin="lower",
    extent=[
        wf_freq[0] / 1e6,
        wf_freq[-1] / 1e6,
        0,
        len(blocks) * waterfall_nfft / SAMPLE_RATE
    ]
)

plt.xlabel("Frequency [MHz]")
plt.ylabel("Time [s]")
plt.title("Waterfall around 99.9 MHz")
plt.colorbar(label="Magnitude [dB]")

plt.tight_layout()

plt.show()