from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AxisProfile:
    name: str
    travel_mm: float
    steps_per_mm: float
    pul_pin: int
    dir_pin: int
    ena_pin: int
    current_a: float
    motors: int
    enable_active_high: bool


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    lead_screw_pitch_mm_per_rev: float
    steps_per_rev: int
    microsteps: int
    axes: dict[str, AxisProfile]

    @property
    def steps_per_mm(self) -> float:
        return (self.steps_per_rev * self.microsteps) / self.lead_screw_pitch_mm_per_rev


def load_hardware_profile(path: str | Path) -> HardwareProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    machine = data["machine"]
    drivers: dict[str, Any] = data["drivers"]
    travel = machine["travel_mm"]
    calculated_steps_per_mm = (
        float(machine["steps_per_rev"]) * float(machine["microsteps"])
    ) / float(machine["lead_screw_pitch_mm_per_rev"])

    axes = {
        axis: AxisProfile(
            name=axis,
            travel_mm=float(travel[axis]),
            steps_per_mm=calculated_steps_per_mm,
            pul_pin=int(driver["pul_pin"]),
            dir_pin=int(driver["dir_pin"]),
            ena_pin=int(driver["ena_pin"]),
            current_a=float(driver["current_a"]),
            motors=int(driver["motors"]),
            enable_active_high=bool(driver["enable_active_high"]),
        )
        for axis, driver in drivers.items()
    }

    return HardwareProfile(
        name=machine["name"],
        lead_screw_pitch_mm_per_rev=float(machine["lead_screw_pitch_mm_per_rev"]),
        steps_per_rev=int(machine["steps_per_rev"]),
        microsteps=int(machine["microsteps"]),
        axes=axes,
    )


def mm_to_steps(distance_mm: float, steps_per_mm: float) -> int:
    return int(round(distance_mm * steps_per_mm))
