"""
CostCalculator unit test
Tests consistency between Python and C++ implementations
"""
from StateConstraint import StateConstraint
from ControlConstraint import ControlConstraint
from state_cost import StateCost as OriginalStateCost
from obstacle_base import obstacle
from kinematic_mode import KinematicModel
from CostCalculator import CostCalculator
import numpy as np

# Parameters matching C++ implementation
DT = 0.1
HORIZON_LENGTH = 60  # Must match Python StateCost.horizon = 60
TEST_HORIZON = 5  # For tests 1-4, use first 5 steps only
STATE_DIM = 4
CONTROL_DIM = 2

# Create a wrapper that fixes the horizon bug in Python StateCost
class StateCost(OriginalStateCost):
    def get_ref_states(self, pos):
        """Override to fix hardcoded horizon bug"""
        n_steps = pos.shape[0]  # Use actual size of input
        ref_waypoints_reshaped = self.ref_waypoints.transpose()[:, :, np.newaxis]
        distances = np.sum((pos.T - ref_waypoints_reshaped) ** 2, axis = 1)
        arg_min_dist_indices = np.argmin(distances, axis = 0)
        ref_exact_points = self.ref_waypoints[:, arg_min_dist_indices]

        self.ref_states = np.vstack([
            ref_exact_points,
            np.full(n_steps, self.ref_velo),
            np.zeros(n_steps)
        ]).T

# Vehicle parameters
LENGTH = 4.5
WIDTH = 2.0
WB = 3.6
SAFETY_BUFFER = 1.5

# Velocity limits
V_MAX = 15.0
V_MIN = 0.0

def print_test_header(test_num, test_name):
    print(f"\n{'='*60}")
    print(f"Test Case {test_num}: {test_name}")
    print(f"{'='*60}")

def print_cost_breakdown(state_cost, state_constraints, control_constraints):
    print(f"  State Cost: {state_cost:.6f}")
    print(f"  State Constraints: {state_constraints:.6f}")
    print(f"  Control Constraints: {control_constraints:.6f}")

def print_derivatives_summary(lx, lxx, lu, luu, lxu):
    print(f"  lx shape: {lx.shape}, norm: {np.linalg.norm(lx):.6f}")
    print(f"  lxx shape: {lxx.shape}, norm: {np.linalg.norm(lxx):.6f}")
    print(f"  lu shape: {lu.shape}, norm: {np.linalg.norm(lu):.6f}")
    print(f"  luu shape: {luu.shape}, norm: {np.linalg.norm(luu):.6f}")
    print(f"  lxu shape: {lxu.shape}, norm: {np.linalg.norm(lxu):.6f}")

def test_case_1_minimal_cost():
    """Test Case 1: Minimal Cost (No Constraints Active)"""
    print_test_header(1, "Minimal Cost (No Constraints Active)")

    # Use TEST_HORIZON for easier debugging
    horizon = TEST_HORIZON

    # Create simple trajectory - straight line at constant velocity
    states = np.zeros((horizon + 1, STATE_DIM))
    controls = np.zeros((horizon, CONTROL_DIM))

    for i in range(horizon + 1):
        states[i] = [i * 0.5, 0.0, 5.0, 0.0]  # [x, y, v, yaw]

    # Reference waypoints along the trajectory
    ref_waypoints = np.zeros((2, 1000))  # Use many waypoints like in test.py
    ref_waypoints[0, :] = np.linspace(0, 50, 1000)
    ref_waypoints[1, :] = 0.0

    # No obstacles
    obstacle_list = []

    # Cost matrices
    Q = np.diag([1.0, 1.0, 0.5, 0])
    R = np.diag([1.0, 1.0])

    # Create components
    vehicle = KinematicModel()
    state_cost = StateCost(Q, R, ref_waypoints, STATE_DIM, CONTROL_DIM)
    control_constraint = ControlConstraint(STATE_DIM, CONTROL_DIM)
    state_constraint = StateConstraint(STATE_DIM, CONTROL_DIM, vehicle, obstacle_list)

    cost_calculator = CostCalculator(
        state_cost, state_constraint, control_constraint,
        horizon, STATE_DIM, CONTROL_DIM
    )

    # Test CalculateTotalCost
    total_cost = cost_calculator.CalculateTotalCost(states, controls)
    print(f"Total Cost: {total_cost:.6f}")

    # Test CalculateDerivates
    lx, lxx, lu, luu, lxu = cost_calculator.CalculateDerivates(states, controls)
    print_derivatives_summary(lx, lxx, lu, luu, lxu)

    return total_cost, lx, lxx, lu, luu, lxu, horizon

def test_case_2_velocity_constraint():
    """Test Case 2: Velocity Constraint Violation"""
    print_test_header(2, "Velocity Constraint Violation")

    # Use TEST_HORIZON for easier debugging
    horizon = TEST_HORIZON

    # Create trajectory with high velocity
    states = np.zeros((horizon + 1, STATE_DIM))
    controls = np.zeros((horizon, CONTROL_DIM))

    for i in range(horizon + 1):
        states[i] = [i * 0.5, 0.0, 20.0, 0.0]  # v=20 > V_MAX=15

    # Reference waypoints
    ref_waypoints = np.zeros((2, 1000))
    ref_waypoints[0, :] = np.linspace(0, 50, 1000)
    ref_waypoints[1, :] = 0.0

    # No obstacles
    obstacle_list = []

    # Cost matrices
    Q = np.diag([1.0, 1.0, 0.5, 0])
    R = np.diag([1.0, 1.0])

    # Create components
    vehicle = KinematicModel()
    state_cost = StateCost(Q, R, ref_waypoints, STATE_DIM, CONTROL_DIM)
    control_constraint = ControlConstraint(STATE_DIM, CONTROL_DIM)
    state_constraint = StateConstraint(STATE_DIM, CONTROL_DIM, vehicle, obstacle_list)

    cost_calculator = CostCalculator(
        state_cost, state_constraint, control_constraint,
        horizon, STATE_DIM, CONTROL_DIM
    )

    # Test CalculateTotalCost
    total_cost = cost_calculator.CalculateTotalCost(states, controls)
    print(f"Total Cost: {total_cost:.6f}")

    # Test CalculateDerivates
    lx, lxx, lu, luu, lxu = cost_calculator.CalculateDerivates(states, controls)
    print_derivatives_summary(lx, lxx, lu, luu, lxu)

    return total_cost, lx, lxx, lu, luu, lxu, horizon

def test_case_3_obstacle_constraint():
    """Test Case 3: Obstacle Constraint Active"""
    print_test_header(3, "Obstacle Constraint Active")

    # Use TEST_HORIZON for easier debugging
    horizon = TEST_HORIZON

    # Create trajectory near obstacle
    states = np.zeros((horizon + 1, STATE_DIM))
    controls = np.zeros((horizon, CONTROL_DIM))

    for i in range(horizon + 1):
        states[i] = [6.5 + i * 0.5, 0.0, 5.0, 0.0]  # Near obstacle at x=6.5

    # Reference waypoints
    ref_waypoints = np.zeros((2, 1000))
    ref_waypoints[0, :] = np.linspace(0, 50, 1000)
    ref_waypoints[1, :] = 0.0

    # Obstacle at [6.5, -0.2, 3.0, 0]
    obstacle_attr = np.array([WIDTH, LENGTH, SAFETY_BUFFER])
    obs_1 = [6.5, -0.2, 3.0, 0.]
    obstacle_list = [obstacle(obs_1, obstacle_attr)]

    # Cost matrices
    Q = np.diag([1.0, 1.0, 0.5, 0])
    R = np.diag([1.0, 1.0])

    # Create components
    vehicle = KinematicModel()
    state_cost = StateCost(Q, R, ref_waypoints, STATE_DIM, CONTROL_DIM)
    control_constraint = ControlConstraint(STATE_DIM, CONTROL_DIM)
    state_constraint = StateConstraint(STATE_DIM, CONTROL_DIM, vehicle, obstacle_list)

    cost_calculator = CostCalculator(
        state_cost, state_constraint, control_constraint,
        horizon, STATE_DIM, CONTROL_DIM
    )

    # Test CalculateTotalCost
    total_cost = cost_calculator.CalculateTotalCost(states, controls)
    print(f"Total Cost: {total_cost:.6f}")

    # Test CalculateDerivates
    lx, lxx, lu, luu, lxu = cost_calculator.CalculateDerivates(states, controls)
    print_derivatives_summary(lx, lxx, lu, luu, lxu)

    return total_cost, lx, lxx, lu, luu, lxu, horizon

def test_case_4_control_constraint():
    """Test Case 4: Control Constraint Violation"""
    print_test_header(4, "Control Constraint Violation")

    # Use TEST_HORIZON for easier debugging
    horizon = TEST_HORIZON

    # Create trajectory with valid states
    states = np.zeros((horizon + 1, STATE_DIM))
    controls = np.zeros((horizon, CONTROL_DIM))

    for i in range(horizon + 1):
        states[i] = [i * 0.5, 0.0, 5.0, 0.0]

    # Large control values (acceleration and steering)
    for i in range(horizon):
        controls[i] = [5.0, 1.5]  # Large acceleration and steering

    # Reference waypoints
    ref_waypoints = np.zeros((2, 1000))
    ref_waypoints[0, :] = np.linspace(0, 50, 1000)
    ref_waypoints[1, :] = 0.0

    # No obstacles
    obstacle_list = []

    # Cost matrices
    Q = np.diag([1.0, 1.0, 0.5, 0])
    R = np.diag([1.0, 1.0])

    # Create components
    vehicle = KinematicModel()
    state_cost = StateCost(Q, R, ref_waypoints, STATE_DIM, CONTROL_DIM)
    control_constraint = ControlConstraint(STATE_DIM, CONTROL_DIM)
    state_constraint = StateConstraint(STATE_DIM, CONTROL_DIM, vehicle, obstacle_list)

    cost_calculator = CostCalculator(
        state_cost, state_constraint, control_constraint,
        horizon, STATE_DIM, CONTROL_DIM
    )

    # Test CalculateTotalCost
    total_cost = cost_calculator.CalculateTotalCost(states, controls)
    print(f"Total Cost: {total_cost:.6f}")

    # Test CalculateDerivates
    lx, lxx, lu, luu, lxu = cost_calculator.CalculateDerivates(states, controls)
    print_derivatives_summary(lx, lxx, lu, luu, lxu)

    return total_cost, lx, lxx, lu, luu, lxu, horizon

def test_case_5_full_scenario():
    """Test Case 5: Full Scenario (matching test.py)"""
    print_test_header(5, "Full Scenario (matching test.py)")

    horizon = 60  # Full horizon from test.py

    # Initial ego state
    ego_state = [0., 0., 5.0, 0.]

    # Reference waypoints (same as test.py)
    n_waypoints = 1000
    longit_ref = np.linspace(0, 50, n_waypoints)
    lateral_ref = np.linspace(0, 0, n_waypoints)
    ref_waypoints = np.vstack((longit_ref, lateral_ref))

    # Obstacles (same as test.py)
    obstacle_attr_1 = np.array([WIDTH, LENGTH, SAFETY_BUFFER])
    obstacle_attr_2 = np.array([WIDTH, LENGTH, SAFETY_BUFFER])
    obs_1 = [6.5, -0.2, 3.0, 0.]
    obs_2 = [20, 4, 2.0, 0.]
    obstacle_list = [obstacle(obs_1, obstacle_attr_1), obstacle(obs_2, obstacle_attr_2)]

    # Cost matrices (same as test.py)
    Q = np.diag([1.0, 1.0, 0.5, 0])
    R = np.diag([1.0, 1.0])

    # Create components
    vehicle = KinematicModel()
    state_cost = StateCost(Q, R, ref_waypoints, STATE_DIM, CONTROL_DIM)
    control_constraint = ControlConstraint(STATE_DIM, CONTROL_DIM)
    state_constraint = StateConstraint(STATE_DIM, CONTROL_DIM, vehicle, obstacle_list)

    cost_calculator = CostCalculator(
        state_cost, state_constraint, control_constraint,
        horizon, STATE_DIM, CONTROL_DIM
    )

    # Initialize trajectory (same as test.py)
    init_control = np.zeros((horizon, 2))
    init_x = vehicle.init_traj(ego_state, init_control)

    # Test CalculateTotalCost
    total_cost = cost_calculator.CalculateTotalCost(init_x, init_control)
    print(f"Total Cost: {total_cost:.6f}")
    print(f"  (Expected ~10651.50 from Python test.py)")

    # Test CalculateDerivates
    lx, lxx, lu, luu, lxu = cost_calculator.CalculateDerivates(init_x, init_control)
    print_derivatives_summary(lx, lxx, lu, luu, lxu)

    return total_cost, lx, lxx, lu, luu, lxu, horizon

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CostCalculator Python Unit Test")
    print("="*60)

    results = {}

    # Run all test cases
    results['test1'] = test_case_1_minimal_cost()
    results['test2'] = test_case_2_velocity_constraint()
    results['test3'] = test_case_3_obstacle_constraint()
    results['test4'] = test_case_4_control_constraint()
    results['test5'] = test_case_5_full_scenario()

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Test 1 (Minimal Cost):         Total Cost = {results['test1'][0]:.6f}")
    print(f"Test 2 (Velocity Constraint):  Total Cost = {results['test2'][0]:.6f}")
    print(f"Test 3 (Obstacle Constraint):  Total Cost = {results['test3'][0]:.6f}")
    print(f"Test 4 (Control Constraint):   Total Cost = {results['test4'][0]:.6f}")
    print(f"Test 5 (Full Scenario):        Total Cost = {results['test5'][0]:.6f}")
