from __future__ import annotations

import argparse
import re
import socketserver
import threading
from dataclasses import dataclass, field
from pathlib import Path

from automation.hardware_profile import HardwareProfile, load_hardware_profile


COMMAND_VALUE_RE = re.compile(r"([XYZxyz])\s*(-?\d+(?:\.\d+)?)")


@dataclass
class MachineState:
    profile: HardwareProfile
    lock: threading.Lock = field(default_factory=threading.Lock)
    position_mm: dict[str, float] = field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
    )
    enabled: bool = True

    def status(self) -> str:
        with self.lock:
            x = self.position_mm["x"]
            y = self.position_mm["y"]
            z = self.position_mm["z"]
            enabled = "1" if self.enabled else "0"
        return f"OK STATUS X{x:.2f} Y{y:.2f} Z{z:.2f} ENABLED{enabled}"

    def home(self) -> str:
        with self.lock:
            self.position_mm = {"x": 0.0, "y": 0.0, "z": 0.0}
        return "OK HOME X0.00 Y0.00 Z0.00"

    def set_origin(self) -> str:
        with self.lock:
            self.position_mm = {"x": 0.0, "y": 0.0, "z": 0.0}
        return "OK ORIGIN"

    def move_relative(self, axis: str, distance_mm: float) -> str:
        axis = axis.lower()
        with self.lock:
            next_pos = self.position_mm[axis] + distance_mm
            self._validate_axis_position(axis, next_pos)
            self.position_mm[axis] = next_pos
        return f"OK MOVE {axis.upper()}{next_pos:.2f}"

    def run_g_command(self, command: str) -> str:
        values = {
            axis.lower(): float(value)
            for axis, value in COMMAND_VALUE_RE.findall(command)
        }
        with self.lock:
            next_position = dict(self.position_mm)
            for axis, value in values.items():
                if axis in next_position:
                    self._validate_axis_position(axis, value)
                    next_position[axis] = value
            self.position_mm = next_position
            x = self.position_mm["x"]
            y = self.position_mm["y"]
            z = self.position_mm["z"]
        return f"OK G X{x:.2f} Y{y:.2f} Z{z:.2f}"

    def _validate_axis_position(self, axis: str, value: float) -> None:
        travel = self.profile.axes[axis].travel_mm
        if value < 0 or value > travel:
            raise ValueError(f"{axis.upper()} axis out of range: {value:.2f} mm, allowed 0..{travel:.2f}")


class MotionRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.wfile.write(b"OK SIM READY\n")
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            command = raw.decode("utf-8", errors="replace").strip()
            if not command:
                continue
            response = self.server.handle_command(command)  # type: ignore[attr-defined]
            self.wfile.write((response + "\n").encode("utf-8"))


class MotionSimulatorServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], state: MachineState):
        super().__init__(server_address, MotionRequestHandler)
        self.state = state

    def handle_command(self, command: str) -> str:
        upper = command.upper()
        try:
            if upper == "STATUS":
                return self.state.status()
            if upper == "HOME":
                return self.state.home()
            if upper == "O":
                return self.state.set_origin()
            if upper.startswith("X "):
                return self.state.move_relative("x", float(command.split(maxsplit=1)[1]))
            if upper.startswith("Y "):
                return self.state.move_relative("y", float(command.split(maxsplit=1)[1]))
            if upper.startswith("Z "):
                return self.state.move_relative("z", float(command.split(maxsplit=1)[1]))
            if upper.startswith("G "):
                return self.state.run_g_command(command)
            return f"ERROR UNKNOWN_COMMAND {command}"
        except Exception as exc:
            return f"ERROR {exc}"


def run_server(host: str, port: int, profile_path: Path) -> None:
    profile = load_hardware_profile(profile_path)
    state = MachineState(profile=profile)
    with MotionSimulatorServer((host, port), state) as server:
        print(f"Motion simulator listening on {host}:{port}")
        print(f"Profile: {profile.name}, steps/mm={profile.steps_per_mm:.2f}")
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="TCP simulator for the three-axis motion controller.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/hardware.feibafly2.example.json"),
    )
    args = parser.parse_args()
    run_server(args.host, args.port, args.profile)


if __name__ == "__main__":
    main()
