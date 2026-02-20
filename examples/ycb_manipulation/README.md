# YCB Manipulation Library + Benchmark

This folder provides:

- `library.py`: builds a manipulable-object library for the YCB dataset
  with validated physical parameters and hard-coded grasp profiles.
- `benchmark.py`: headless xArm7 + LEAP benchmark for all YCB objects,
  checking settle/contact stability and repeated grasp/release behavior.

## Run Full Dataset Benchmark

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python examples/ycb_manipulation/benchmark.py --cycles 2 --tune-grasps
```

## Outputs

- `examples/ycb_manipulation/reports/ycb_manipulation_library.json`
  - Object library with per-object mass/inertia/friction/contact and selected hard-coded grasp profile.
- `examples/ycb_manipulation/reports/ycb_validation_report.json`
  - Full pass/fail diagnostics per object.
- `examples/ycb_manipulation/reports/ycb_validation_report.md`
  - Human-readable summary table.

## Benchmark Criteria

Each object is checked for:

- Natural settle under gravity.
- Contact stability (no bad finite-state explosions, bounded penetration).
- Repeated grasp/release interactions over configured cycles.

Pass/fail is recorded per object with reason flags (`settle`, `grasp`, `release`, `contact`).
