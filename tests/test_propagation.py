import numpy as np

from src.passive_radar.propagation import (
    fractional_delay,
    apply_doppler,
)


sample_rate = 1e6

signal = np.ones(1000, dtype=complex)


# Test fractional delay
delay = 137.532

delayed = fractional_delay(signal, delay)

print("Fractional delay test")
print("---------------------")
print("Input length:   ", len(signal))
print("Output length:  ", len(delayed))
print("Requested delay:", delay)


# Test Doppler
doppler = 161.802

shifted = apply_doppler(
    signal,
    doppler,
    sample_rate,
)

print("\nDoppler test")
print("------------")
print("Doppler:", doppler, "Hz")
print("First sample:", shifted[0])
print("Second sample:", shifted[1])