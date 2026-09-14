import numpy as np
import matplotlib.pyplot as plt

from simulations.radar_signal import generate_fm_signal


# ============================================================
# Parameters
# ============================================================

SAMPLE_RATE = 2.4e6
DURATION = 0.01

N = int(SAMPLE_RATE * DURATION)

FREQUENCY_DEVIATION = 75e3
MODULATION_FREQUENCY = 1e3


# ============================================================
# Generate signal
# ============================================================

signal = generate_fm_signal(
    N=N,
    sample_rate=SAMPLE_RATE,
    frequency_deviation=FREQUENCY_DEVIATION,
    modulation_frequency=MODULATION_FREQUENCY,
)


# ============================================================
# Time domain
# ============================================================

time = np.arange(N) / SAMPLE_RATE

plt.figure(figsize=(10, 5))

plt.plot(
    time[:500] * 1e6,
    signal.real[:500],
)

plt.xlabel("Time [µs]")
plt.ylabel("Amplitude")
plt.title("Synthetic FM Signal - Time Domain")
plt.grid()

plt.show()


# ============================================================
# Spectrum
# ============================================================

window = np.hanning(N)

spectrum = np.fft.fftshift(
    np.fft.fft(signal * window)
)

frequencies = np.fft.fftshift(
    np.fft.fftfreq(N, 1 / SAMPLE_RATE)
)

spectrum_db = 20 * np.log10(
    np.abs(spectrum) / np.max(np.abs(spectrum))
    + 1e-12
)


plt.figure(figsize=(10, 5))

plt.plot(
    frequencies / 1e3,
    spectrum_db,
)

plt.xlim(-200, 200)
plt.ylim(-80, 5)

plt.xlabel("Frequency [kHz]")
plt.ylabel("Magnitude [dB]")
plt.title("Synthetic FM Signal - Spectrum")
plt.grid()

plt.show()