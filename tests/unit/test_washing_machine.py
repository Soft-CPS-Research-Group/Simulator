import unittest
import tempfile
from pathlib import Path
from citylearn.base import EpisodeTracker
from citylearn.data import WashingMachineSimulation
from citylearn.energy_model import WashingMachine
import numpy as np
from unittest.mock import MagicMock, patch

class TestWashingMachine(unittest.TestCase):

    def setUp(self):
        # Create a comprehensive mock for WashingMachineSimulation
        self.sim = MagicMock()
        self.sim.wm_start_time_step = np.array([0, 3, 6, -1, -1])  # -1 indicates no cycle
        self.sim.wm_end_time_step = np.array([2, 5, 8, -1, -1])
        self.sim.load_profile = {
            0: [0.1, 0.2, 0.1],    # Normal cycle
            3: [0.3, 0.4],         # Shorter cycle
            6: [],                 # Empty cycle
            -1: None               # No cycle case
        }

        # Mock for episode tracker
        self.episode_tracker = MagicMock()
        self.episode_tracker.episode_time_steps = 10  # Extended for more test cases

        # Create test machine
        self.machine = WashingMachine(
            washing_machine_simulation=self.sim,
            name="Washer1",
            episode_tracker=self.episode_tracker,
            nominal_power=2.0
        )
        
        # Initialize arrays
        self.machine._WashingMachine__past_action_values = np.zeros(10)
        self.machine._ElectricDevice__electricity_consumption = np.zeros(10)

    def test_no_cycle_time_step(self):
        """Test time steps with no defined cycle"""
        self.machine.time_step = 4  # No cycle at this time
        self.machine.start_cycle(1.0)
        
        self.assertFalse(self.machine.initiated)
        self.assertEqual(self.machine.past_action_values[4], 1.0)

    # Edge Cases
    def test_negative_action_value(self):
        """Test with negative action value (should be handled gracefully)"""
        self.machine.time_step = 0
        self.machine.start_cycle(-0.5)
        
        self.assertFalse(self.machine.initiated)
        self.assertEqual(self.machine.past_action_values[0], -0.5)

    def test_zero_action_value(self):
        """Test with zero action value"""
        self.machine.time_step = 0
        self.machine.start_cycle(0.0)
        
        self.assertFalse(self.machine.initiated)
        self.assertEqual(self.machine.past_action_values[0], 0.0)

    # Observation and State Tests
    def test_observation_structure(self):
        """Test complete observation structure"""
        self.machine.time_step = 0
        self.machine.start_cycle(1.0)
        
        obs = self.machine.observations()
        expected_keys = {
            'washing_machine_initiated',
            'washing_machine_action',
            'wm_start_time_step',
            'wm_end_time_step'
        }
        self.assertTrue(expected_keys.issubset(obs.keys()))
        self.assertTrue(obs['washing_machine_initiated'])
        self.assertEqual(obs['washing_machine_action'], 1.0)

    def test_reset_functionality(self):
        """Test complete reset of all state variables"""
        # Set up some state
        self.machine._WashingMachine__initiated = True
        self.machine._WashingMachine__past_action_values = np.ones(10)
        self.machine._ElectricDevice__electricity_consumption = np.ones(10) * 0.5
        self.machine.time_step = 5
        
        self.machine.reset()
        
        # Verify reset
        self.assertFalse(self.machine.initiated)
        np.testing.assert_array_equal(self.machine.past_action_values, np.zeros(10))
        np.testing.assert_array_equal(self.machine._ElectricDevice__electricity_consumption, np.zeros(10))
        self.assertEqual(self.machine.time_step, 0)

    def test_absolute_cycle_window_aligns_after_nonzero_episode_start(self):
        tracker = EpisodeTracker(0, 7)
        tracker.next_episode(4, rolling_episode_split=False, random_episode_split=False, random_seed=0)
        tracker.next_episode(4, rolling_episode_split=False, random_episode_split=False, random_seed=0)

        sim = WashingMachineSimulation(
            day_type=[1] * 8,
            hour=list(range(8)),
            wm_start_time_step=[1, 1, 1, 1, 5, 5, 5, 5],
            wm_end_time_step=[3, 3, 3, 3, 7, 7, 7, 7],
            load_profile=["[0.5]"] * 8,
        )
        sim.start_time_step = tracker.episode_start_time_step
        sim.end_time_step = tracker.episode_end_time_step
        machine = WashingMachine(
            washing_machine_simulation=sim,
            name="Washer2",
            episode_tracker=tracker,
            nominal_power=2.0,
        )
        machine.reset()
        machine.next_time_step()
        machine.start_cycle(1.0)

        self.assertTrue(machine.initiated)
        self.assertEqual(machine.electricity_consumption[1], 0.5)

    def test_load_profile_parser_does_not_execute_strings(self):
        marker = Path(tempfile.gettempdir()) / "citylearn_eval_should_not_run"
        marker.unlink(missing_ok=True)
        sim = WashingMachineSimulation(
            day_type=[1, 1, 1],
            hour=[1, 2, 3],
            wm_start_time_step=[0, 1, -1],
            wm_end_time_step=[0, 1, -1],
            load_profile=[
                "[0.1, 0.2]",
                f"__import__('pathlib').Path({str(marker)!r}).write_text('x')",
                "-1",
            ],
        )

        np.testing.assert_allclose(sim.load_profile[0], np.array([0.1, 0.2]))
        self.assertEqual(len(sim.load_profile[1]), 0)
        self.assertEqual(len(sim.load_profile[2]), 0)
        self.assertFalse(marker.exists())

    def test_repeated_load_profiles_reuse_parsed_array_instances(self):
        sim = WashingMachineSimulation(
            day_type=[1, 1, 1, 1],
            hour=[1, 2, 3, 4],
            wm_start_time_step=[0, 0, -1, -1],
            wm_end_time_step=[1, 1, -1, -1],
            load_profile=["[0.1, 0.2]", "[0.1, 0.2]", "-1", ""],
        )

        self.assertIs(sim.load_profile[0], sim.load_profile[1])
        self.assertIs(sim.load_profile[2], sim.load_profile[3])

    # Exception Handling
    def test_invalid_time_step(self):
        """Test behavior with invalid time steps"""
        with self.assertRaises(IndexError):
            self.machine.time_step = 20  # Beyond episode bounds
            self.machine.start_cycle(1.0)

    @patch('numpy.zeros')
    def test_memory_allocation_failure(self, mock_zeros):
        """Test handling of array initialization failure"""
        mock_zeros.side_effect = MemoryError("Out of memory")
        with self.assertRaises(MemoryError):
            bad_machine = WashingMachine(
                washing_machine_simulation=self.sim,
                episode_tracker=self.episode_tracker
            )
            bad_machine.next_time_step()

if __name__ == "__main__":
    unittest.main()
