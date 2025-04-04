import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from citylearn.electric_vehicle_charger import Charger
from citylearn.base import EpisodeTracker
from citylearn.electric_vehicle import ElectricVehicle
from citylearn.energy_model import Battery

@pytest.fixture
def mock_episode_tracker():
    tracker = MagicMock(spec=EpisodeTracker)
    tracker.episode_time_steps = 24
    return tracker

@pytest.fixture
def mock_battery():
    battery = MagicMock(spec=Battery)
    battery.capacity = 100.0  # 100 kWh battery
    battery.nominal_power = 10.0  # 10 kW nominal power
    battery.initial_soc = 0.5  # Start at 50% SOC
    battery.soc = np.array([0.5] * 24)  # SOC time series
    battery.energy_balance = np.zeros(24)  # Track energy changes
    battery.efficiency = 0.9  # 90% efficiency
    battery.round_trip_efficiency = 0.9**0.5  # sqrt(efficiency)
    battery.time_step = 0
    
    def charge_side_effect(energy_kwh):
        nonlocal battery
        if battery.time_step == 0:
            prev_soc = battery.initial_soc
        else:
            prev_soc = battery.soc[battery.time_step - 1]
        
        if energy_kwh >= 0:  # Charging
            effective_energy = energy_kwh * battery.round_trip_efficiency
        else:  # Discharging
            effective_energy = energy_kwh / battery.round_trip_efficiency
        
        new_soc = (prev_soc * battery.capacity + effective_energy) / battery.capacity
        new_soc = max(0.0, min(1.0, new_soc))
        
        battery.soc[battery.time_step] = new_soc
        battery.energy_balance[battery.time_step] = effective_energy
    
    battery.charge.side_effect = charge_side_effect
    return battery

@pytest.fixture
def mock_electric_vehicle(mock_battery):
    ev = MagicMock(spec=ElectricVehicle)
    ev.battery = mock_battery
    ev.name = "TestEV"
    ev.electric_vehicle_simulation = MagicMock()
    ev.electric_vehicle_simulation.electric_vehicle_charger_state = np.array([1] * 24)
    ev.electric_vehicle_simulation.electric_vehicle_required_soc_departure = np.array([0.8] * 24)
    ev.electric_vehicle_simulation.electric_vehicle_estimated_soc_arrival = np.array([0.4] * 24)
    ev.electric_vehicle_simulation.electric_vehicle_estimated_arrival_time = np.array([0] * 24)
    ev.electric_vehicle_simulation.electric_vehicle_departure_time = np.array([23] * 24)
    return ev

@pytest.fixture
def charger(mock_episode_tracker):
    charger = Charger(
        episode_tracker=mock_episode_tracker,
        charger_id="TestCharger",
        max_charging_power=10.0,
        min_charging_power=1.0,
        max_discharging_power=10.0,
        min_discharging_power=1.0,
        connected_electric_vehicle=None
    )
    charger._Charger__electricity_consumption = np.zeros(24)
    charger._Charger__past_charging_action_values_kwh = np.zeros(24)
    charger._Charger__past_connected_evs = [None] * 24
    charger.time_step = 0
    charger.seconds_per_time_step = 3600  # 1 hour
    yield charger
    charger.connected_electric_vehicle = None
    charger.incoming_electric_vehicle = None

# --- BASIC FUNCTIONALITY TESTS ---

def test_initialization(charger):
    """Test that charger initializes with correct default values"""
    assert charger.charger_id == "TestCharger"
    assert charger.max_charging_power == 10.0
    assert charger.min_charging_power == 1.0
    assert charger.max_discharging_power == 10.0
    assert charger.min_discharging_power == 1.0
    assert charger.efficiency == 1.0
    assert charger.connected_electric_vehicle is None
    assert charger.incoming_electric_vehicle is None

def test_plug_car(charger, mock_electric_vehicle):
    """Test plugging in an electric vehicle"""
    charger.plug_car(mock_electric_vehicle)
    assert charger.connected_electric_vehicle == mock_electric_vehicle
    assert charger._Charger__past_connected_evs[0] == mock_electric_vehicle

def test_plug_car_when_occupied(charger, mock_electric_vehicle):
    """Test that plugging fails when charger is already occupied"""
    charger.plug_car(mock_electric_vehicle)
    with pytest.raises(ValueError):
        charger.plug_car(mock_electric_vehicle)

def test_associate_incoming_car(charger, mock_electric_vehicle):
    """Test associating an incoming electric vehicle"""
    charger.associate_incoming_car(mock_electric_vehicle)
    assert charger.incoming_electric_vehicle == mock_electric_vehicle

# --- ACTION AND POWER TESTS ---

def test_action_value_storage(charger, mock_electric_vehicle):
    """Verify charger stores past energies correctly"""
    charger.connected_electric_vehicle = mock_electric_vehicle
    
    test_actions = [0.3, -0.5, 0.0, 1.0]
    max_charging_power = 10
    for i, action in enumerate(test_actions):
        charger.time_step = i
        charger.update_connected_electric_vehicle_soc(action)
        energy = action * max_charging_power 
        assert charger._Charger__past_charging_action_values_kwh[i] == energy

def test_power_clamping(charger, mock_electric_vehicle):
    """Verify power values are clamped correctly"""
    charger.connected_electric_vehicle = mock_electric_vehicle
    
    # Test cases: (action, expected_power_kw)
    test_cases = [
        (0.5, 5.0),    # 50% of max charging
        (1.5, 10.0),    # Above max -> clamp to max
        (0.05, 1.0),    # Below min -> clamp to min
        (-0.5, -5.0),   # 50% discharging
        (-1.5, -10.0),  # Max discharging
        (-0.05, -1.0)   # Min discharging
    ]
    
    for action, expected_power in test_cases:
        charger.update_connected_electric_vehicle_soc(action)
        args, _ = mock_electric_vehicle.battery.charge.call_args
        actual_power_kw = args[0]  # Energy in kWh is same as power in kW for 1 hour
        assert actual_power_kw == pytest.approx(expected_power)
        mock_electric_vehicle.battery.charge.reset_mock()

def test_electricity_consumption_calculation(charger, mock_electric_vehicle): #TODO: Rever isto
    # TODO
    assert True == True

def test_no_action_when_no_ev(charger):
    """Verify no action is taken when no EV is connected"""
    action = 0.5
    max_charging_power = 10
    charger.update_connected_electric_vehicle_soc(action)
    energy = action * max_charging_power 
    assert charger._Charger__electricity_consumption[0] == 0
    assert charger._Charger__past_charging_action_values_kwh[0] == energy

def test_zero_action(charger, mock_electric_vehicle):
    """Verify zero action results in no energy exchange"""
    charger.connected_electric_vehicle = mock_electric_vehicle
    charger.update_connected_electric_vehicle_soc(0.0)
    
    assert charger._Charger__electricity_consumption[0] == 0
    assert not mock_electric_vehicle.battery.charge.called

# --- EFFICIENCY CURVE TESTS ---

def test_efficiency_curve_interpolation():
    """Test efficiency curve interpolation"""
    episode_tracker = MagicMock(spec=EpisodeTracker)
    episode_tracker.episode_time_steps = 24
    
    # Create charger with properly shaped efficiency curves
    charger = Charger(
        episode_tracker=episode_tracker,
        charge_efficiency_curve=[[0, 0.83],[0.3, 0.83],[0.7, 0.9],[0.8, 0.9],[1, 0.85]],  # [power_levels], [efficiencies]
        discharge_efficiency_curve=[[0, 0.63],[0.3, 0.23],[0.7, 0.5],[0.8, 0.4],[1, 0.75]]
    )
    
    # Verify the curves were properly initialized
    assert charger.charge_efficiency_curve.shape == (2, 5)
    assert charger.discharge_efficiency_curve.shape == (2, 5)
    
    # Test charging efficiency
    assert charger.get_efficiency(0.0, True) == 0.83
    assert charger.get_efficiency(0.7, True) == 0.9
    assert charger.get_efficiency(1.0, True) == 0.85
    assert charger.get_efficiency(0.5, True) == pytest.approx(0.865)  # interpolated
    
    # Test discharging efficiency
    assert charger.get_efficiency(0.0, False) == 0.63
    assert charger.get_efficiency(0.7, False) == 0.5
    assert charger.get_efficiency(1.0, False) == 0.75
    assert charger.get_efficiency(0.5, False) == pytest.approx(0.365)  # interpolated

def test_default_efficiency_without_curve(charger):
    """Test that default efficiency is used when no curve is provided"""
    assert charger.get_efficiency(0.5, True) == 1.0
    assert charger.get_efficiency(0.5, False) == 1.0

# --- TIME STEP AND RESET TESTS ---

def test_next_time_step(charger, mock_electric_vehicle):
    """Test reset functionality"""
    charger.plug_car(mock_electric_vehicle)
    charger.associate_incoming_car(mock_electric_vehicle)
    print(mock_electric_vehicle)

    assert charger.connected_electric_vehicle is not None
    assert charger.incoming_electric_vehicle is not None
    
    charger.time_step = 3

    charger.next_time_step()

    assert charger.time_step == 4
    assert charger.connected_electric_vehicle is None
    assert charger.incoming_electric_vehicle is None

    


def test_reset(charger, mock_electric_vehicle):
    """Test reset functionality"""
    charger.plug_car(mock_electric_vehicle)
    charger.update_connected_electric_vehicle_soc(0.5)
    
    charger.time_step = 3

    charger.reset()
    
    assert charger.time_step == 0
    assert charger.connected_electric_vehicle is None
    assert charger.incoming_electric_vehicle is None
    assert all(ec == 0 for ec in charger._Charger__electricity_consumption)
    assert all(pc == 0 for pc in charger._Charger__past_charging_action_values_kwh)
    assert all(ev is None for ev in charger._Charger__past_connected_evs)
