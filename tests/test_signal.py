import numpy as np

from src.passive_radar.signal import (
    generate_reference_signal,
    generate_noise,
)


N = 10_000


reference = generate_reference_signal(N, seed=42)
noise = generate_noise(N, amplitude=0.5, seed=42)


print("Signal test")
print("-----------")

print("Reference samples:", len(reference))
print("Noise samples:    ", len(noise))

print("Reference mean:   ", np.mean(reference))
print("Noise mean:       ", np.mean(noise))

print("Reference power:  ", np.mean(np.abs(reference) ** 2))
print("Noise power:      ", np.mean(np.abs(noise) ** 2))