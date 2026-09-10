import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Parameters
# ============================================================

CENTER_FREQUENCY = 100e6
SAMPLE_RATE = 2.4e6

INPUT_FILE = "recording.npy"


# ============================================================
# Load IQ recording
# ============================================================

print()
print("=" * 60)
print("RTL-SDR RECORDING ANALYSIS")
print("=" * 60)

samples = np.load(INPUT_FILE)

samples = np.asarray(samples, dtype=np.complex64)

N = len(samples)

print(f"Samples:       {N:,}")
print(f"Sample rate:   {SAMPLE_RATE / 1e6:.3f} MS/s")
print(f"Center freq:   {CENTER_FREQUENCY / 1e6:.3f} MHz")
print(f"Duration:      {N / SAMPLE_RATE:.3f} s")
print(f"Data type:     {samples.dtype}")
print(f"File data:     {samples.nbytes / 1e6:.1f} MB")


# ============================================================
# Basic statistics
# ============================================================

mean = np.mean(samples)
rms = np.sqrt(np.mean(np.abs(samples) ** 2))
peak = np.max(np.abs(samples))

print()
print("SIGNAL STATISTICS")
print("-" * 60)

print(f"Mean:          {mean}")
print(f"RMS amplitude: {rms:.6f}")
print(f"Peak amplitude:{peak:.6f}")


# ============================================================
# Plot IQ signal
# ============================================================

# Only plot the first 100,000 samples
plot_samples = min(100_000, N)

time = np.arange(plot_samples) / SAMPLE_RATE

plt.figure(figsize=(12, 5))

plt.plot(
    time * 1e3,
    samples[:plot_samples].real,
    label="I"
)

plt.plot(
    time * 1e3,
    samples[:plot_samples].imag,
    label="Q"
)

plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")

plt.title("IQ Signal")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ============================================================
# FFT / Spectrum
# ============================================================

# Use a manageable section for the FFT
fft_samples = min(2**20, N)

signal = samples[:fft_samples]

window = np.hanning(fft_samples)

spectrum = np.fft.fftshift(
    np.fft.fft(signal * window)
)

spectrum = np.abs(spectrum)

# Convert to dB
spectrum_db = 20 * np.log10(
    spectrum / np.max(spectrum) + 1e-12
)

frequencies = np.fft.fftshift(
    np.fft.fftfreq(
        fft_samples,
        d=1 / SAMPLE_RATE
    )
)

# Convert baseband frequency to RF frequency
rf_frequencies = (
    CENTER_FREQUENCY
    + frequencies
)


# ============================================================
# Plot spectrum
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    rf_frequencies / 1e6,
    spectrum_db
)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Relative amplitude (dB)")

plt.title("RTL-SDR Spectrum")

plt.grid()

plt.ylim(-100, 5)

plt.tight_layout()
plt.show()


# ============================================================
# Find strongest frequencies
# ============================================================

# Ignore very small frequencies around the edges
valid = spectrum_db > -40

strongest_indices = np.argsort(
    spectrum[valid]
)[-10:]

valid_indices = np.where(valid)[0]

strongest_indices = valid_indices[
    strongest_indices
]

strongest_indices = strongest_indices[
    np.argsort(
        spectrum[strongest_indices]
    )[::-1]
]


print()
print("STRONGEST FREQUENCIES")
print("-" * 60)

for index in strongest_indices:

    frequency = rf_frequencies[index] / 1e6
    level = spectrum_db[index]

    print(
        f"{frequency:.6f} MHz    "
        f"{level:.2f} dB"
    )


# ============================================================
# Waterfall
# ============================================================

waterfall_samples = min(
    2**20,
    N
)

waterfall_signal = samples[:waterfall_samples]

nfft = 4096

num_rows = waterfall_samples // nfft

waterfall_signal = waterfall_signal[
    :num_rows * nfft
]

waterfall_signal = waterfall_signal.reshape(
    num_rows,
    nfft
)

window = np.hanning(nfft)

waterfall_fft = np.fft.fftshift(
    np.fft.fft(
        waterfall_signal * window,
        axis=1
    ),
    axes=1
)

waterfall_power = np.abs(
    waterfall_fft
)

waterfall_db = 20 * np.log10(
    waterfall_power
    / np.max(waterfall_power)
    + 1e-12
)

waterfall_frequencies = np.fft.fftshift(
    np.fft.fftfreq(
        nfft,
        d=1 / SAMPLE_RATE
    )
)

waterfall_rf = (
    CENTER_FREQUENCY
    + waterfall_frequencies
)

waterfall_time = (
    np.arange(num_rows)
    * nfft
    / SAMPLE_RATE
)


# ============================================================
# Plot waterfall
# ============================================================

plt.figure(figsize=(12, 7))

plt.imshow(
    waterfall_db,
    aspect="auto",
    origin="lower",
    extent=[
        waterfall_rf[0] / 1e6,
        waterfall_rf[-1] / 1e6,
        waterfall_time[0],
        waterfall_time[-1]
    ]
)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Time (s)")

plt.title("RTL-SDR Waterfall")

plt.colorbar(
    label="Relative amplitude (dB)"
)

plt.clim(-50, 0)

plt.tight_layout()
plt.show()