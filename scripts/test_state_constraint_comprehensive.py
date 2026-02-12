"""
Comprehensive StateConstraint Verification Test
Tests StateConstraint implementation against known values for comparison with C++.
Contains 10 test scenarios covering velocity constraints and obstacle avoidance.
"""

import numpy as np
import sys
sys.path.append("/home/yuhao/Code/Panda_ilqr/scripts")

from StateConstraint import StateConstraint
from kinematic_mode import KinematicModel
from obstacle_base import obstacle

def print_array(name, arr, precision=10):
    """Print array with specified precision"""
    if arr.ndim == 1:
        print(f"{name}: [{', '.join(f'{x:.{precision}f}' for x in arr)}]")
    else:
        print(f"{name}:")
        for row in arr:
            print(f"  [{', '.join(f'{x:.{precision}f}' for x in row)}]")

def run_test_scenario(name, state, obstacle_states, obstacle_attrs, steps_to_test=[0, 30, 59]):
    """Run a single test scenario and print results"""
    print(f"\n{'='*70}")
    print(f"Scenario: {name}")
    print(f"{'='*70}")
    print(f"State: [{', '.join(f'{x:.10f}' for x in state)}]")

    # Create obstacles
    obstacles = []
    for i, (obs_state, obs_attr) in enumerate(zip(obstacle_states, obstacle_attrs)):
        obs_state_arr = np.array(obs_state)
        obs_attr_arr = np.array(obs_attr)
        obs = obstacle(obs_state_arr, obs_attr_arr)
        obstacles.append(obs)
        print(f"Obstacle {i+1}: state=[{', '.join(f'{x:.10f}' for x in obs_state)}], "
              f"attr=[{', '.join(f'{x:.10f}' for x in obs_attr)}], "
              f"a={obs.a:.10f}, b={obs.b:.10f}")

    if not obstacles:
        print("No obstacles")

    # Create StateConstraint
    model = KinematicModel()
    constraint = StateConstraint(4, 2, model, obstacles)

    # Test value() at different steps
    print(f"\n--- value() results ---")
    val = 0.0  # Store last value for return
    for step in steps_to_test:
        control = np.array([0.0, 0.0])  # Dummy control, not used in StateConstraint
        val = constraint.value(step, state, control)
        print(f"Step {step:2d}: value = {val:.10f}")

    # Test gradient_lx()
    print(f"\n--- gradient_lx() results at step 0 ---")
    control = np.array([0.0, 0.0])
    lx = constraint.gradient_lx(0, state, control)
    print(f"lx = [{', '.join(f'{x:.10f}' for x in lx)}]")

    # Test hessian_lxx()
    print(f"\n--- hessian_lxx() results at step 0 ---")
    control = np.array([0.0, 0.0])
    lxx = constraint.hessian_lxx(0, state, control)
    print(f"lxx =")
    print_array("lxx", lxx)

    return {
        'value': val if obstacles else 0.0,
        'gradient': lx,
        'hessian': lxx
    }

def main():
    print("="*70)
    print("Comprehensive StateConstraint Verification Test - 10 Scenarios")
    print("="*70)

    # Scenario 1: No Obstacle, Mid Velocity
    print("\n" + "="*70)
    print("Scenario 1: No Obstacle, Mid Velocity")
    print("="*70)
    state1 = np.array([5.0, 3.0, 5.0, 0.0])
    obs_states1 = []
    obs_attrs1 = []
    results1 = run_test_scenario("01: No Obstacle, Mid Velocity",
                            state1, obs_states1, obs_attrs1)

    # Scenario 2: Single Obstacle, Safe Distance
    state2 = np.array([5.0, 3.0, 5.0, 0.0])
    obs_states2 = [[10.0, 5.0, 2.0, 0.0]]
    obs_attrs2 = [[2.0, 4.5, 1.5]]
    results2 = run_test_scenario("02: Single Obstacle, Safe Distance",
                            state2, obs_states2, obs_attrs2)

    # Scenario 3: Single Obstacle, Near Collision
    state3 = np.array([8.0, 4.5, 5.0, 0.0])
    obs_states3 = [[10.0, 5.0, 2.0, 0.0]]
    obs_attrs3 = [[2.0, 4.5, 1.5]]
    results3 = run_test_scenario("03: Single Obstacle, Near Collision",
                            state3, obs_states3, obs_attrs3)

    # Scenario 4: Single Obstacle, Collision Front
    state4 = np.array([10.0, 5.0, 5.0, 0.0])
    obs_states4 = [[10.0, 5.0, 2.0, 0.0]]
    obs_attrs4 = [[2.0, 4.5, 1.5]]
    results4 = run_test_scenario("04: Single Obstacle, Collision Front",
                            state4, obs_states4, obs_attrs4)

    # Scenario 5: Single Obstacle, Collision Rear
    state5 = np.array([10.0, 5.0, 5.0, 0.0])
    obs_states5 = [[10.0, 5.0, 2.0, 0.0]]
    obs_attrs5 = [[2.0, 4.5, 1.5]]
    results5 = run_test_scenario("05: Single Obstacle, Collision Rear",
                            state5, obs_states5, obs_attrs5)

    # Scenario 6: Single Obstacle, High Velocity
    state6 = np.array([5.0, 3.0, 10.0, 0.0])
    obs_states6 = [[10.0, 5.0, 2.0, 0.0]]
    obs_attrs6 = [[2.0, 4.5, 1.5]]
    results6 = run_test_scenario("06: Single Obstacle, High Velocity",
                            state6, obs_states6, obs_attrs6)

    # Scenario 7: Single Obstacle, Low Velocity
    state7 = np.array([5.0, 3.0, 0.5, 0.0])
    obs_states7 = [[10.0, 5.0, 2.0, 0.0]]
    obs_attrs7 = [[2.0, 4.5, 1.5]]
    results7 = run_test_scenario("07: Single Obstacle, Low Velocity",
                            state7, obs_states7, obs_attrs7)

    # Scenario 8: Two Obstacles, Different Yaws
    state8 = np.array([0.0, 0.0, 5.0, 0.0])
    obs_states8 = [
        [6.5, -0.2, 3.0, 0.0],      # yaw = 0
        [20.0, 4.0, 2.0, np.pi/4]   # yaw = 45°
    ]
    obs_attrs8 = [
        [2.0, 4.5, 1.5],
        [2.0, 4.5, 1.5]
    ]
    results8 = run_test_scenario("08: Two Obstacles, Different Yaws",
                            state8, obs_states8, obs_attrs8)

    # Scenario 9: Moving Obstacle, Various Steps
    state9 = np.array([0.0, 0.0, 5.0, 0.0])
    obs_states9 = [[6.5, -0.2, 3.0, 0.5]]
    obs_attrs9 = [[2.0, 4.5, 1.5]]
    results9 = run_test_scenario("09: Moving Obstacle, Various Steps",
                            state9, obs_states9, obs_attrs9,
                            steps_to_test=[0, 10, 30, 50, 59])

    # Scenario 10: Edge Case - Zero Velocity
    state10 = np.array([5.0, 3.0, 0.0, 0.0])
    obs_states10 = [[10.0, 5.0, 2.0, 0.0]]
    obs_attrs10 = [[2.0, 4.5, 1.5]]
    results10 = run_test_scenario("10: Edge Case - Zero Velocity",
                             state10, obs_states10, obs_attrs10)

    print("\n" + "="*70)
    print("All 10 scenarios completed!")
    print("="*70)

if __name__ == "__main__":
    main()
