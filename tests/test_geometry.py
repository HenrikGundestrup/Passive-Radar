from src.passive_radar.geometry import (
    distance,
    bistatic_range,
    bistatic_delay,
    bistatic_doppler,
)


tx = (0, 0)
rx = (10_000, 0)
target = (5_000, 20_000)

velocity = (0, 250)

carrier_frequency = 100e6


print("TX -> RX distance:")
print(distance(tx, rx), "m")

print("\nBistatic range:")
print(bistatic_range(tx, target, rx), "m")

print("\nBistatic delay:")
print(bistatic_delay(tx, target, rx), "s")

print("\nBistatic Doppler:")
print(
    bistatic_doppler(
        tx,
        target,
        rx,
        velocity,
        carrier_frequency,
    ),
    "Hz",
)