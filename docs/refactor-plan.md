# Refactor Plan

The original prototype is mostly contained in `main.py`. That made early iteration fast, but it now mixes UI, hardware control, OCR, AI calls, task scheduling, and configuration in one file.

## Current Extraction

The repository now has a small standalone utility layer:

- `automation/hardware_profile.py`: loads hardware parameters and calculates axis movement constants.
- `automation/simulator.py`: TCP simulator for the motion controller.
- `scripts/simulate_link_check.py`: starts the simulator and verifies the PC-to-controller command link.
- `configs/hardware.feibafly2.example.json`: sanitized hardware profile.
- `docs/hardware-setup.md`: wiring, travel, and safety notes.
- `firmware/arduino_uno_motion_controller/arduino_uno_motion_controller.ino`: Arduino firmware draft matching the current pinout.

## Recommended Next Split

1. Move `ESPController` into `automation/controller_client.py`.
2. Move screenshot capture into `automation/screenshot.py`.
3. Move OCR and image processing into `automation/vision.py`.
4. Move AI decision parsing into `automation/ai_planner.py`.
5. Move `Task`, `TaskExecutor`, and `TaskScheduler` into `automation/tasks/`.
6. Keep PyQt widgets under `ui/`.
7. Add unit tests for config loading, command parsing, coordinate mapping, and scheduler state transitions.

## Why This Order

The motion-control link is the riskiest part because it touches hardware. Extracting and simulating that boundary first lets future UI/OCR/AI changes be tested without moving real motors.

