import numpy as np

from src.passive_radar.signal import generate_reference_signal
from src.passive_radar.cancellation import (
    estimate_direct_path,
    cancel_direct_path,
)


N = 10_000
delay = 50
amplitude = 10.0

reference = generate_reference_signal(N, seed=42)

delayed_reference = np.zeros_like(reference)
delayed_reference[delay:] = reference[:-delay]

surveillance = amplitude * delayed_reference

estimated = estimate_direct_path(
    reference,
    surveillance,
    delay,
)

cancelled = cancel_direct_path(
    reference,
    surveillance,
    delay,
    estimated,
)

print("Cancellation test")
print("------------------")

print("True coefficient:     ", amplitude)
print("Estimated coefficient:", estimated)

print(
    "Before cancellation:",
    np.mean(np.abs(surveillance) ** 2),
)

print(
    "After cancellation: ",
    np.mean(np.abs(cancelled) ** 2),
)