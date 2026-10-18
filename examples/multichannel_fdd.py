"""Identify the modes of a three-story shear building from synthetic sensors."""

from pathlib import Path
import tempfile

import numpy as np

import timoshenko as tm

building = tm.Structure([12_000.0, 12_000.0, 12_000.0], [8.0e6, 8.0e6, 8.0e6])
model_freqs = building.natural_frequencies_hz
print("Model frequencies (Hz):", [round(freq, 3) for freq in model_freqs])

# Analytical mode shapes of a uniform shear building; columns are modes,
# rows are floors from the base upward. Shape k is sin(pi * floor *
# (k + 0.5) / (stories + 1)).
floors = np.arange(1.0, 4.0)
model_shapes = np.sin(np.pi * floors[:, None] * np.arange(0.5, 3.0, 1.0) / 4.0)
# Unequal weights make the noise-robustness point clearer: a weak middle
# mode still has to be found, which is why nperseg is chosen below.
weights = np.array([1.0, 0.4, 2.0])

sampling_hz = 128.0
sample_count = 16_384
time = np.arange(sample_count) / sampling_hz
rng = np.random.default_rng(7)
modal_responses = np.column_stack(
    [
        np.sin(2.0 * np.pi * freq * time + phase)
        for freq, phase in zip(model_freqs, (0.0, 0.3, 1.1), strict=True)
    ]
)
# Response per channel: each floor sees the sum of all modes scaled by the
# mode's amplitude weight and its story amplitude in the shape.
channels = modal_responses @ (weights[:, None] * model_shapes.T)
channels += 0.02 * rng.standard_normal(channels.shape)

observations = tm.MultiChannelData(
    channels,
    sampling_hz=sampling_hz,
    channel_ids=("floor_1", "floor_2", "floor_3"),
    units=("m/s^2", "m/s^2", "m/s^2"),
)
print("Channels:", observations.channel_ids)

# A window of 2048 samples gives a 0.0625 Hz resolution with enough averages
# to resolve all three modes of this record.
fdd = tm.identify_fdd(observations, nperseg=2048, max_modes=3)
print(f"Spectral resolution: {fdd.resolution_hz:.4f} Hz")
for mode, expected in zip(fdd.modes, model_shapes.T, strict=False):
    shape = np.asarray(mode.shape_real, dtype=float)
    mac = tm.modal_assurance_criterion(expected, shape)
    print(f"Mode {mode.frequency_hz:.3f} Hz (ratio {mode.singular_value_ratio:.3f}), MAC {mac:.3f}")
    print("  shape:", np.round(shape, 3))

# The same channels could also travel through a CSV round trip.
with tempfile.TemporaryDirectory() as directory:
    csv_path = Path(directory) / "channels.csv"
    header = ",".join(observations.channel_ids)
    rows = [
        ",".join(f"{value:.17g}" for value in observations.samples[row])
        for row in range(sample_count)
    ]
    csv_path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    reloaded = tm.load_multichannel_csv(
        csv_path, columns=list(observations.channel_ids), sampling_hz=sampling_hz
    )
print("CSV round trip samples:", reloaded.sample_count)
