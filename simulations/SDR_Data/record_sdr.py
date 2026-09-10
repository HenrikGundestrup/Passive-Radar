import numpy as np
from rtlsdr import RtlSdr
import time


# ============================================================
# Parameters
# ============================================================

CENTER_FREQUENCY = 100e6      # 100 MHz
SAMPLE_RATE = 2.4e6           # 2.4 MS/s
GAIN = 30                     # dB
RECORD_TIME = 10              # seconds

OUTPUT_FILE = "recording.npy"


# ============================================================
# Calculate number of samples
# ============================================================

num_samples = int(SAMPLE_RATE * RECORD_TIME)

print()
print("=" * 60)
print("RTL-SDR RECORDING")
print("=" * 60)

print(f"Center frequency: {CENTER_FREQUENCY / 1e6:.3f} MHz")
print(f"Sample rate:      {SAMPLE_RATE / 1e6:.3f} MS/s")
print(f"Recording time:   {RECORD_TIME:.1f} s")
print(f"Samples:          {num_samples:,}")
print()


# ============================================================
# Connect to SDR
# ============================================================

sdr = RtlSdr()

sdr.sample_rate = SAMPLE_RATE
sdr.center_freq = CENTER_FREQUENCY
sdr.gain = GAIN


# ============================================================
# Record
# ============================================================

print("Starting recording...")

start_time = time.time()

samples = sdr.read_samples(num_samples)

elapsed = time.time() - start_time

print("Recording finished.")
print(f"Actual recording time: {elapsed:.2f} s")


# ============================================================
# Close SDR
# ============================================================

sdr.close()


# ============================================================
# Save IQ data
# ============================================================

samples = np.asarray(samples, dtype=np.complex64)

np.save(
    OUTPUT_FILE,
    samples
)

print()
print(f"Saved to: {OUTPUT_FILE}")
print(f"Samples saved: {len(samples):,}")
print(f"Data type: {samples.dtype}")
print(f"File size: {samples.nbytes / 1e6:.1f} MB")