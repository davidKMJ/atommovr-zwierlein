# AWG Controller

Hardware control package for driving Spectrum Instrumentation AWG cards with DDS (Direct Digital Synthesis) during atom rearrangement.

## Overview

This package isolates all AWG / DDS control logic from the `atommovr` simulation framework. It translates logical atom moves into RF frequency ramps executed by the card's DDS cores, driving acousto-optic deflectors (AODs) that physically reposition trapped atoms.

```mermaid
flowchart LR
    Cam["Camera<br/>(real / offline)"] -->|occupancy| Ctrl["atommovrController"]
    Algo["Rearrangement<br/>Algorithm"] -->|Move list| RF["RFConverter<br/>(awg_control)"]
    Ctrl --> Algo
    RF -->|AWGBatch| Strat["DDS Strategy<br/>(dds_strategies)"]
    Strat -->|spcm API| Card["AWG Card<br/>(hardware)"]
    Card -->|RF signal| AOD["AOD"]
```

## Directory Structure

```
awg_controller/
├── src/
│   ├── __init__.py          # Re-exports public API
│   ├── awg_control.py       # RFConverter, AODSettings, AWGBatch, hardware constants
│   ├── dds_strategies.py    # 4 DDS strategy classes + registry
│   ├── camera.py            # Camera ABC, RealArrayCamera, OfflineArrayCamera
│   ├── offline_camera.py    # Compatibility shim → camera.py
│   └── session_recorder.py  # Optional stage dumps / rounds.jsonl / GIFs
├── scripts/
│   ├── atommover_controller.py  # Closed-loop feedback controller
│   └── atommovr_controller.py   # Compatibility shim → atommover_controller
├── tests/
│   ├── test_awg_control.py
│   ├── test_controller_pipeline.py
│   ├── test_offline_and_calibration.py
│   └── test_session_recorder.py
├── docs/
│   ├── dds_strategy_ramp.md
│   ├── dds_strategy_pattern.md
│   └── dds_strategy_camera_triggered.md
├── config/
└── README.md
```

## Hardware

### Spectrum Instrumentation AWG Card

- **DDS cores**: 21 total (indices 0-20)
- **Channel 0** (V / row AOD): cores 0-7, 12-19 (exclusive); cores 8-11 (flex)
- **Channel 1** (H / col AOD): cores 8-11 (flex) + core 20 (fixed)
- **Maximum tones**: 20 on ch0 (single ch1) or 16 on ch0 + 5 on ch1 (dual ch1)
- **Amplitude budget**: 40 % of full-scale per channel, distributed equally across active tones

### What to Look Out For

> **Output amplitude MUST stay below 2.0 V at all times.**
> The default `HardwareConfig.max_amplitude_v` is 1.6 V.
> **Exceeding 2.0 V will permanently damage the AOD amplifier.**

- **Always test with an oscilloscope** before connecting the amplifier to the AOD.
- Start with `max_amplitude_v = 1.0` (conservative) and increase gradually.
- The camera-triggered strategy enforces `trigger_level_v < 2.0 V` with a hard `ValueError`.
- Use `validate_hardware_limits(grid_rows, grid_cols)` at startup to fail fast if grid dimensions exceed DDS core capacity.

## DDS Strategies

Four interchangeable strategies implement the `DDSStrategy` interface.

**Timing**: AWG move pacing uses **travel** only —
`atommovr.utils.timing.travel_duration_s` (Chebyshev × spacing / `AOD_speed`).
`HardwareConfig.trigger_timer_s` / `--trg-timer` is the idle/holding TIMER.
Camera TTL is independent of travel.

| Strategy           | Trigger                  | Frequency transition | Key advantage                           |
| ------------------ | ------------------------ | -------------------- | --------------------------------------- |
| `streaming`        | TIMER                    | Abrupt hop           | Simple, battle-tested                   |
| `ramp`             | TIMER + FPGA slope       | Smooth sweep         | Best transport quality, S-curve support |
| `pattern`          | TIMER + CARD + `force()` | Abrupt hop           | No FIFO underrun risk                   |
| `camera_triggered` | TIMER + CARD + ext0 TTL  | Abrupt hop           | Zero software jitter, hardware-synced   |

Select a strategy by name or instance:

```python
from awg_controller.src.dds_strategies import get_strategy

strategy = get_strategy("ramp", use_scurve=True, scurve_segments=16)
```

Detailed documentation for each strategy is in `docs/`.

## Camera and recording

`Camera` (`src/camera.py`) is the shared imaging interface: acquire →
`detect_occupancy` → `sync(array, recorder=…)`. Subclasses:

- `OfflineArrayCamera` — synthetic fluorescence for closed-loop tests without hardware
- `RealArrayCamera` — wraps a grabber callable for a physical camera

`SessionRecorder` (`src/session_recorder.py`) is optional; pass it to the
controller (`recorder=`) for per-stage dumps, `rounds.jsonl`, and GIFs.

## Usage

### RF Conversion (no hardware needed)

```python
from awg_controller.src.awg_control import RFConverter, AODSettings
from atommovr.utils.core import PhysicalParams
from atommovr.utils.Move import Move

settings = AODSettings(
    f_min_v=60e6, f_max_v=100e6,
    f_min_h=60e6, f_max_h=100e6,
    grid_rows=10, grid_cols=5,
)
converter = RFConverter(settings, PhysicalParams())

# Convert moves to RF commands
batch = converter.convert_moves([Move(0, 0, 2, 1)])
print(f"{len(batch.ramps)} ramps, travel={batch.total_duration_s*1e6:.1f} µs")
```

### Full Controller (simulation mode)

```python
from awg_controller.scripts.atommover_controller import (
    atommovrController, HardwareConfig, SoftwareConfig,
)
from atommovr.utils.core import PhysicalParams

sw = SoftwareConfig(
    grid_size=10,
    algorithm_name="PCFA",
    physical_params=PhysicalParams(middle_size=[6, 6]),
)
hw = HardwareConfig(trigger_timer_s=0.2)  # idle / holding TIMER

with atommovrController(sw, hw, strategy="ramp") as ctrl:
    success = ctrl.run()
```

Defaults to an `OfflineArrayCamera` when no camera is passed. Attach a
`SessionRecorder` via `recorder=` for stage dumps and `rounds.jsonl`.

### Full Controller (with hardware)

```python
hw = HardwareConfig(
    card_paths=["/dev/spcm0"],
    max_amplitude_v=1.6,        # NEVER exceed 2.0 V
    trigger_timer_s=0.2,
)
sw = SoftwareConfig(
    grid_size=10,
    algorithm_name="PCFA",
    physical_params=PhysicalParams(middle_size=[6, 5]),
    aod_settings=AODSettings(
        f_min_v=60e6, f_max_v=100e6,
        f_min_h=60e6, f_max_h=100e6,
        grid_rows=10, grid_cols=5,
    ),
)

with atommovrController(sw, hw, strategy="pattern") as ctrl:
    success = ctrl.run()
```

### CLI

```bash
python awg_controller/scripts/atommover_controller.py \
    --algorithm PCFA \
    --grid-rows 10 --grid-cols 5 \
    --target-rows 6 --target-cols 5 \
    --strategy ramp \
    --trg-timer 0.2
```

## Testing

Run all AWG controller tests (no hardware required):

```bash
pytest awg_controller/tests/ -v
```

## spcm Documentation

This package builds on the [spcm Python driver](https://github.com/SpectrumInstrumentation/spcm) for Spectrum Instrumentation cards. Key references:

- [spcm GitHub repository](https://github.com/SpectrumInstrumentation/spcm)
- [spcm DDS examples](https://github.com/SpectrumInstrumentation/spcm/tree/master/src/examples) — examples 03, 04, 09, 12, 15 form the basis of the four strategies
- [Spectrum documentation portal](https://spectruminstrumentation.github.io/spcm/spcm.html)

## Dependencies

- **Runtime**: `numpy`, `spcm` (optional — simulation mode works without it)
- **Algorithms & imaging**: `atommovr.algorithms`, `atommovr.utils` (from the parent repo)
- **Tests**: `pytest`, `numpy`
