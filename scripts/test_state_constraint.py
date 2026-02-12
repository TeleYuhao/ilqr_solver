"""
StateConstraint Verification Test
Tests the StateConstraint implementation against known values for comparison with C++.
"""

import numpy as np
import sys
sys.path.append("/home/yuhao/Code/Panda_ilqr/scripts")

from cost_base import CostFunc
from obstacle_base import obstacle
from kinematic_mode import KinematicModel
from StateConstraint import StateConstraint

class config:
    v_max = 10
    v_min = 0
    a_max = 2.0
    a_min = -2.0
    delta_max = 1.57
    delta_min = -1.57

def print_array(name, arr, precision=10):
    """Print array with specified precision"""
    if arr.ndim == 1:
        print(f"{name}: [{', '.join(f'{x:.{precision}f}' for x in arr)}]")
    else:
        print(f"{name}:")
        for row in arr:
            print(f"  [{', '.join(f'{x:.{precision}f}' for x in row)}]")

def run_test_case(name, state, control, obstacle_list, step, model):
    """Run a single test case and print results"""
    print(f"\n{'='*60}")
    print(f"Test Case: {name}")
    print(f"{'='*60}")
    print(f"State: {state}")
    print(f"Control: {control}")
    print(f"Step: {step}")
    print(f"Obstacle state: {obstacle_list[0].state}")
    print(f"Obstacle attr: {obstacle_list[0].attr}")

    # Create StateConstraint
    state_constraint = StateConstraint(4, 2, model, obstacle_list)

    # Test value()
    value = state_constraint.value(step, state, control)
    print(f"\nvalue(): {value:.10f}")

    # Test gradient_lx()
    lx = state_constraint.gradient_lx(step, state, control)
    print_array("gradient_lx()", lx)

    # Test hessian_lxx()
    lxx = state_constraint.hessian_lxx(step, state, control)
    print_array("hessian_lxx()", lxx)

    # Test gradient_lu()
    lu = state_constraint.gradient_lu(step, state, control)
    print_array("gradient_lu()", lu)

    # Test hessian_luu()
    luu = state_constraint.hessian_luu(step, state, control)
    print_array("hessian_luu()", luu)

    # Test hessian_lxu()
    lxu = state_constraint.hessian_lxu(step, state, control)
    print_array("hessian_lxu()", lxu)

    return {
        'value': value,
        'lx': lx,
        'lxx': lxx,
        'lu': lu,
        'luu': luu,
        'lxu': lxu
    }

def main():
    print("="*60)
    print("StateConstraint Verification Test")
    print("="*60)

    # Create kinematic model
    model = KinematicModel()

    # Test Case 1: State Near Obstacle
    print("\n" + "="*60)
    print("Test Case 1: State Near Obstacle")
    print("="*60)
    state1 = np.array([5.0, 2.0, 5.0, 0.5])  # [x, y, v, yaw]
    control1 = np.array([0.5, 0.1])  # [a, delta]
    # Create obstacle near the state
    obs1_state = np.array([10.0, 5.0, 2.0, 0.0])  # [x, y, v, yaw]
    obs1_attr = np.array([2.0, 4.5, 1.5])  # [width, length, d_safe]
    obs1 = obstacle(obs1_state, obs1_attr)
    results1 = run_test_case("Near Obstacle", state1, control1, [obs1], 0, model)

    # Test Case 2: Velocity Lower Bound Violation
    state2 = np.array([0.0, 0.0, -0.5, 0.0])  # v < V_MIN
    control2 = np.array([0.0, 0.0])
    # Create far away obstacle (should not affect)
    obs2_state = np.array([50.0, 50.0, 0.0, 0.0])
    obs2_attr = np.array([2.0, 4.5, 1.5])
    obs2 = obstacle(obs2_state, obs2_attr)
    results2 = run_test_case("Velocity Below V_MIN", state2, control2, [obs2], 0, model)

    # Test Case 3: Velocity Upper Bound Violation
    state3 = np.array([0.0, 0.0, 11.0, 0.0])  # v > V_MAX
    control3 = np.array([0.0, 0.0])
    obs3 = obstacle(obs2_state, obs2_attr)  # Same far obstacle
    results3 = run_test_case("Velocity Above V_MAX", state3, control3, [obs3], 0, model)

    # Test Case 4: Velocity Within Bounds (no obstacle cost)
    state4 = np.array([0.0, 0.0, 5.0, 0.0])  # v in [V_MIN, V_MAX]
    control4 = np.array([0.0, 0.0])
    obs4 = obstacle(obs2_state, obs2_attr)
    results4 = run_test_case("Velocity Within Bounds", state4, control4, [obs4], 0, model)

    # Test Case 5: Combined - Near obstacle with velocity at boundary
    state5 = np.array([8.0, 4.0, 10.0, 0.0])  # v = V_MAX
    control5 = np.array([0.0, 0.0])
    # Obstacle closer to the state
    obs5_state = np.array([12.0, 6.0, 2.0, 0.0])
    obs5_attr = np.array([2.0, 4.5, 1.5])
    obs5 = obstacle(obs5_state, obs5_attr)
    results5 = run_test_case("Combined: V_MAX + Near Obstacle", state5, control5, [obs5], 0, model)

    # Test Case 6: Yaw angle effect (non-zero yaw)
    state6 = np.array([5.0, 2.0, 5.0, 1.0])  # yaw = 1.0 rad
    control6 = np.array([0.0, 0.0])
    obs6 = obstacle(obs1_state, obs1_attr)
    results6 = run_test_case("Non-zero Yaw (1.0 rad)", state6, control6, [obs6], 0, model)

    # Test Case 7: Different step index (obstacle prediction)
    state7 = np.array([0.0, 0.0, 5.0, 0.0])
    control7 = np.array([0.0, 0.0])
    # Obstacle moving toward origin
    obs7_state = np.array([15.0, 5.0, 2.0, -0.5])  # Moving with yaw = -0.5
    obs7_attr = np.array([2.0, 4.5, 1.5])
    obs7 = obstacle(obs7_state, obs7_attr)
    # Use step 10 to see obstacle prediction effect
    results7 = run_test_case("Step 10 (Obstacle Prediction)", state7, control7, [obs7], 10, model)

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

    return {
        'test1': results1,
        'test2': results2,
        'test3': results3,
        'test4': results4,
        'test5': results5,
        'test6': results6,
        'test7': results7
    }

if __name__ == "__main__":
    results = main()
