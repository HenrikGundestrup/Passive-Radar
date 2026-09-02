import numpy as np

from src.passive_radar.signal import generate_reference_signal
from src.passive_radar.processing import cross_correlate


N = 10_000
true_delay = 500

reference = generate_reference_signal(
    N,
    seed=42,
)

surveillance = np.zeros_like(reference)

surveillance[true_delay:] = reference[:-true_delay]


correlation, lags = cross_correlate(
    reference,
    surveillance,
)

peak_index = np.argmax(np.abs(correlation))
detected_delay = lags[peak_index]


print("Cross-correlation test")
print("----------------------")

print("True delay:     ", true_delay)
print("Detected delay: ", detected_delay)