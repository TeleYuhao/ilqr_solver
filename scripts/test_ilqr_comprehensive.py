"""
Comprehensive ILQR Solver Verification Test
Tests ILQR_Core implementation against known values for comparison with C++.
Tests backward_pass, forward_pass, iter, and solve methods.
"""

import numpy as np
import sys
sys.path.append("/home/yuhao/Code/Panda_ilqr/scripts")

from ILQR_Core import ilqr
from CostCalculator import CostCalculator
from StateCost import StateCost
from StateConstraint import StateConstraint
from ControlConstraint import ControlConstraint
from obstacle_base import obstacle
from kinematic_mode import KinematicModel

def print_array(name, arr, precision=8):
    """Print array with specified precision"""
    if arr.ndim == 1:
        print(f"{name}: [{', '.join(f'{x:.{precision}f}' for x in arr)}]")
    else:
        print(f"{name}:")
        for row in arr:
            print(f"  [{', '.join(f'{x:.{precision}f}' for x in row)}]")

def print_matrix(name, mat, precision=8):
    """Print matrix with specified precision"""
    print(f"{name}:")
    for row in mat:
        print(f"  [{', '.join(f'{x:.{precision}f}' for x in row)}]")

def setup_test_scenario():
    """Setup the same test scenario as scripts/test.py"""
    # Initial ego state: [x, y, v, yaw]
    ego_state = np.array([0.0, 0.0, 5.0, 0.0])

    # Reference waypoints (straight line along x-axis, 0 to 50)
    n_waypoints = 1000
    ref_waypoints = np.zeros((2, n_waypoints))
    ref_waypoints[0, :] = np.linspace(0, 50, n_waypoints)
    ref_waypoints[1, :] = 0.0

    # Weight matrices
    Q = np.diag([1.0, 1.0, 0.5, 0.0])
    R = np.diag([1.0, 1.0])

    # Vehicle parameters
    WIDTH = 2.0
    LENGTH = 4.5
    SAFETY_BUFFER = 1.5

    # Obstacles
    obstacle_attr_1 = np.array([WIDTH, LENGTH, SAFETY_BUFFER])
    obstacle_attr_2 = np.array([WIDTH, LENGTH, SAFETY_BUFFER])
    obs_1 = np.array([6.5, -0.2, 3.0, 0.0])
    obs_2 = np.array([20.0, 4.0, 2.0, 0.0])

    obstacle_list = [obstacle(obs_1, obstacle_attr_1), obstacle(obs_2, obstacle_attr_2)]

    # Create components
    state_cost = StateCost(Q, R, ref_waypoints, 4, 2)
    control_constraint = ControlConstraint(4, 2)
    vehicle = KinematicModel()
    state_constraint = StateConstraint(4, 2, vehicle, obstacle_list)

    # Create cost calculator
    cost_calculator = CostCalculator(state_cost, state_constraint, control_constraint, 60, 4, 2)

    # Initialize trajectory with zero controls
    init_control = np.zeros((60, 2))
    init_x = vehicle.init_traj(ego_state, init_control)

    # Update reference states in state cost
    cost_calculator.StateCost.get_ref_states(init_x[:, :2])

    # Get initial cost
    init_J = cost_calculator.CalculateTotalCost(init_x, init_control)

    # Create solver
    ilqr_solver = ilqr(vehicle, cost_calculator)

    return ego_state, init_x, init_control, init_J, ilqr_solver

def test_backward_pass():
    """Test 1: backward_pass() at Iteration 0"""
    print("=" * 70)
    print("Test 1: backward_pass() at Iteration 0")
    print("=" * 70)

    ego_state, x, u, J, solver = setup_test_scenario()
    lamb = 20.0

    print(f"\nInitial cost: {J:.8f}")
    print(f"Lambda: {lamb:.8f}")
    print(f"Horizon: {solver.N}")

    # Call backward_pass
    d, K, exp_redu = solver.backward_pass(u, x, lamb)

    print(f"\nExpected cost reduction: {exp_redu:.8f}")
    print(f"\n--- Feedforward gains d (last 5 steps) ---")
    print_array("d[59]", d[59])
    print_array("d[58]", d[58])
    print_array("d[57]", d[57])
    print_array("d[56]", d[56])
    print_array("d[55]", d[55])

    print(f"\n--- Feedback gains K (last 3 steps) ---")
    print_matrix("K[59]", K[59])
    print_matrix("K[58]", K[58])
    print_matrix("K[57]", K[57])

    return d, K

def test_forward_pass(d, K):
    """Test 2: forward_pass() with Known Gains"""
    print("\n" + "=" * 70)
    print("Test 2: forward_pass() with Known Gains")
    print("=" * 70)

    ego_state, x, u, J, solver = setup_test_scenario()

    print(f"\nInitial cost: {J:.8f}")
    print(f"\n--- Testing alpha options ---")

    alpha_options = [1.0, 0.5, 0.25, 0.125, 0.0625]

    for alpha in alpha_options:
        new_u, new_x = solver.forward_pass(u, x, d, K, alpha)

        # Calculate new cost
        new_J = solver.CostFunc.CalculateTotalCost(new_x, new_u)

        print(f"\nAlpha = {alpha:.4f}")
        print(f"  New cost: {new_J:.8f}")
        print(f"  Cost reduction: {J - new_J:.8f}")

        # Print first few states
        print(f"  First state: [{new_x[0][0]:.6f}, {new_x[0][1]:.6f}, {new_x[0][2]:.6f}, {new_x[0][3]:.6f}]")
        print(f"  State at step 10: [{new_x[10][0]:.6f}, {new_x[10][1]:.6f}, {new_x[10][2]:.6f}, {new_x[10][3]:.6f}]")

def test_iter():
    """Test 3: iter() Single Iteration"""
    print("\n" + "=" * 70)
    print("Test 3: iter() Single Iteration")
    print("=" * 70)

    ego_state, x, u, J, solver = setup_test_scenario()
    lamb = 20.0

    print(f"\nInitial cost: {J:.8f}")
    print(f"Lambda: {lamb:.8f}")

    # Run single iteration
    new_u, new_x, new_J, effective = solver.iter(u, x, J, lamb)

    print(f"\nNew cost: {new_J:.8f}")
    print(f"Cost reduction: {J - new_J:.8f}")
    print(f"Effective: {effective}")

    print(f"\n--- First few new states ---")
    for i in [0, 10, 20, 30]:
        print(f"Step {i:2d}: [{new_x[i][0]:.6f}, {new_x[i][1]:.6f}, {new_x[i][2]:.6f}, {new_x[i][3]:.6f}]")

def test_solve():
    """Test 4: Full solve() Convergence"""
    print("\n" + "=" * 70)
    print("Test 4: Full solve() Convergence")
    print("=" * 70)

    ego_state, x, u, J, solver = setup_test_scenario()

    print(f"\nInitial state: [{ego_state[0]:.1f}, {ego_state[1]:.1f}, {ego_state[2]:.1f}, {ego_state[3]:.1f}]")
    print(f"Initial cost: {J:.8f}")
    print(f"Max iterations: {solver.max_iter}")
    print(f"Tolerance: {solver.tol}")

    # Run full optimization
    u_opt, x_opt = solver.solve(ego_state)

    # Print final trajectory summary
    print(f"\n--- Final Trajectory (every 10 steps) ---")
    for i in range(0, 61, 10):
        print(f"Step {i:2d}: [{x_opt[i][0]:.6f}, {x_opt[i][1]:.6f}, {x_opt[i][2]:.6f}, {x_opt[i][3]:.6f}]")

    # Calculate final cost
    final_J = solver.CostFunc.CalculateTotalCost(x_opt, u_opt)
    print(f"\nFinal cost: {final_J:.8f}")

    return u_opt, x_opt, final_J

def main():
    print("=" * 70)
    print("Comprehensive ILQR Solver Verification Test")
    print("=" * 70)

    # Run all tests
    d, K = test_backward_pass()
    test_forward_pass(d, K)
    test_iter()
    u_opt, x_opt, final_J = test_solve()

    print("\n" + "=" * 70)
    print("All ILQR solver tests completed!")
    print("=" * 70)

    # Final verification
    print(f"\nExpected final cost: ~429")
    print(f"Actual final cost: {final_J:.2f}")

    if abs(final_J - 429) < 10:
        print("SUCCESS: Convergence matches expected!")
    else:
        print("WARNING: Convergence differs from expected!")

if __name__ == "__main__":
    main()
