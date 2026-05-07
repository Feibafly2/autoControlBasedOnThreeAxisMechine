from pathlib import Path
import unittest

from automation.hardware_profile import load_hardware_profile, mm_to_steps


class HardwareProfileTests(unittest.TestCase):
    def test_feibafly2_profile_calculates_steps_per_mm(self) -> None:
        profile = load_hardware_profile(Path("configs/hardware.feibafly2.example.json"))

        self.assertEqual(profile.steps_per_mm, 200.0)
        self.assertEqual(profile.axes["x"].travel_mm, 300.0)
        self.assertEqual(profile.axes["y"].motors, 2)
        self.assertEqual(profile.axes["z"].ena_pin, 10)
        self.assertEqual(mm_to_steps(12.5, profile.steps_per_mm), 2500)


if __name__ == "__main__":
    unittest.main()
