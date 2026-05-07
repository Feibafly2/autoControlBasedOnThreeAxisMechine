from pathlib import Path
import unittest

from automation.hardware_profile import load_hardware_profile
from automation.simulator import MachineState


def make_state() -> MachineState:
    profile = load_hardware_profile(Path("configs/hardware.feibafly2.example.json"))
    return MachineState(profile=profile)


class SimulatorStateTests(unittest.TestCase):
    def test_g_command_updates_absolute_position(self) -> None:
        state = make_state()

        response = state.run_g_command("G X10 Y20 Z5 M1")

        self.assertEqual(response, "OK G X10.00 Y20.00 Z5.00")
        self.assertEqual(state.status(), "OK STATUS X10.00 Y20.00 Z5.00 ENABLED1")


    def test_axis_bounds_are_enforced(self) -> None:
        state = make_state()

        with self.assertRaisesRegex(ValueError, "X axis out of range"):
            state.run_g_command("G X301 Y0 Z0")


if __name__ == "__main__":
    unittest.main()
