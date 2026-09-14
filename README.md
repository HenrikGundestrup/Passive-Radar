# Passive Radar

A Python-based simulation of a passive bistatic radar system.

## Project overview

This project investigates the principles and signal processing techniques used in passive bistatic radar.

The current simulation models:

- Bistatic radar geometry
- Transmitter-to-target-to-receiver propagation
- Direct-path propagation
- Target propagation delay
- Doppler shift
- Complex baseband signals
- Fractional sample delays
- Direct-path estimation and cancellation
- Cross-correlation
- Slow-time Doppler processing
- Range-Doppler maps

## Current simulation

The current simulation generates a synthetic reference signal and uses it to simulate both:

1. A strong direct-path signal between transmitter and receiver
2. A weaker signal reflected from a moving target

The surveillance signal is then processed to estimate the direct-path signal and suppress it before performing delay-Doppler processing.

## Example scenario

The current simulation uses a bistatic geometry with:

- Carrier frequency: 100 MHz
- Sample rate: 1 MHz
- Transmitter-receiver separation: 10 km
- Target position: 5 km × 20 km
- Target velocity: 250 m/s
- Direct-path amplitude: 10
- Target amplitude: 0.1
- Noise amplitude: 0.5

## Project structure

```text
Passive-Radar/
│
├── README.md
├── requirements.txt
│
├── simulations/
│   └── basic_simulation.py
│
├── src/
├── tests/
└── docs/
                 ┌──────────────────┐
                 │  Generate FM     │
                 │    reference     │
                 └────────┬─────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
       Direct path              Target path
       delay = 33.356           delay = 137.532
              │                       │
              │                 + Doppler
              │                 -161.80 Hz
              │                       │
              └───────────┬───────────┘
                          ▼
                  + receiver noise
                          │
                          ▼
                    Surveillance
                          │
                          ▼
               Direct-path estimation
                          │
                          ▼
                Direct-path cancellation
                          │
                          ▼
                 Range-Doppler processing
                          │
                          ▼
                     Target peak