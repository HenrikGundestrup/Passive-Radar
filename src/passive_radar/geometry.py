import numpy as np


C = 299_792_458.0


def distance(p1, p2):
    """Calculate Euclidean distance between two 2D points."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    return np.linalg.norm(p2 - p1)


def bistatic_range(tx, target, rx):
    """Calculate the bistatic path length TX -> target -> RX."""
    return distance(tx, target) + distance(target, rx)


def bistatic_delay(tx, target, rx):
    """Calculate bistatic propagation delay."""
    return bistatic_range(tx, target, rx) / C


def bistatic_doppler(
    tx,
    target,
    rx,
    target_velocity,
    carrier_frequency,
):
    """
    Calculate bistatic Doppler frequency.

    The Doppler shift is determined from the target velocity
    projected onto the TX-target and target-RX directions.
    """

    tx = np.asarray(tx, dtype=float)
    target = np.asarray(target, dtype=float)
    rx = np.asarray(rx, dtype=float)
    velocity = np.asarray(target_velocity, dtype=float)

    # Unit vector from TX to target
    u_tx = (target - tx) / np.linalg.norm(target - tx)

    # Unit vector from target to RX
    u_rx = (rx - target) / np.linalg.norm(rx - target)

    # Rate of change of bistatic path length
    range_rate = np.dot(velocity, u_tx - u_rx)

    wavelength = C / carrier_frequency

    return -range_rate / wavelength