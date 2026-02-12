"""
Comprehensive Obstacle Verification Test
Tests Obstacle implementation against known values for comparison with C++.
Contains 15 test cases covering basic functionality, edge cases, and boundary conditions.
"""

import numpy as np
import sys
sys.path.append("/home/yuhao/Code/Panda_ilqr/scripts")

from obstacle_base import obstacle

def print_array(name, arr, precision=10):
    """Print array with specified precision"""
    if arr.ndim == 1:
        print(f"{name}: [{', '.join(f'{x:.{precision}f}' for x in arr)}]")
    else:
        print(f"{name}:")
        for row in arr:
            print(f"  [{', '.join(f'{x:.{precision}f}' for x in row)}]")

def run_test_case(name, obs_state, obs_attr, test_point=None, elp_center=None):
    """Run a single test case and print results"""
    print(f"\n{'='*70}")
    print(f"Test Case: {name}")
    print(f"{'='*70}")
    print(f"Obstacle state: [{', '.join(f'{x:.10f}' for x in obs_state)}]")
    print(f"Obstacle attr: [{', '.join(f'{x:.10f}' for x in obs_attr)}]")

    # Create obstacle
    obs = obstacle(obs_state, obs_attr)

    # Print ellipsoid parameters
    print(f"\nEllipsoid semi-axes:")
    print(f"  a (semi-major): {obs.a:.10f}")
    print(f"  b (semi-minor): {obs.b:.10f}")

    # Test safety margin if points provided
    if test_point is not None and elp_center is not None:
        print(f"\nTest point: [{', '.join(f'{x:.10f}' for x in test_point)}]")
        print(f"Ellipsoid center: [{', '.join(f'{x:.10f}' for x in elp_center)}]")

        safety_margin = obs.ellipsoid_safety_margin(test_point, elp_center)
        print(f"\nellipsoid_safety_margin(): {safety_margin:.10f}")

        derivatives = obs.ellipsoid_safety_margin_derivatives(test_point, elp_center)
        print_array("ellipsoid_safety_margin_derivatives():", derivatives)

    # Print prediction trajectory info
    pred_traj = obs.prediction_traj
    print(f"\nPrediction trajectory:")
    print(f"  Shape: ({pred_traj.shape[0]}, {pred_traj.shape[1]})")
    print(f"  Initial state: [{', '.join(f'{x:.10f}' for x in pred_traj[0])}]")
    print(f"  Final state: [{', '.join(f'{x:.10f}' for x in pred_traj[-1])}]")

    # Print intermediate states
    print(f"  State at step 10: [{', '.join(f'{x:.10f}' for x in pred_traj[10])}]")
    print(f"  State at step 30: [{', '.join(f'{x:.10f}' for x in pred_traj[30])}]")
    print(f"  State at step 50: [{', '.join(f'{x:.10f}' for x in pred_traj[50])}]")

    return {
        'a': obs.a,
        'b': obs.b,
        'safety_margin': safety_margin if test_point is not None else None,
        'derivatives': derivatives if test_point is not None else None,
        'pred_traj_shape': pred_traj.shape
    }

def run_grid_test_case(name, obs_state, obs_attr, elp_center):
    """Run a grid of test points around obstacle"""
    print(f"\n{'='*70}")
    print(f"Test Case: {name}")
    print(f"{'='*70}")
    print(f"Obstacle state: [{', '.join(f'{x:.10f}' for x in obs_state)}]")
    print(f"Obstacle attr: [{', '.join(f'{x:.10f}' for x in obs_attr)}]")
    print(f"Ellipsoid center: [{', '.join(f'{x:.10f}' for x in elp_center)}]")

    # Create obstacle
    obs = obstacle(obs_state, obs_attr)

    print(f"\nEllipsoid semi-axes:")
    print(f"  a (semi-major): {obs.a:.10f}")
    print(f"  b (semi-minor): {obs.b:.10f}")

    # Create 3x3 grid of test points
    cx, cy = elp_center
    offsets = [-5.0, 0.0, 5.0]

    print(f"\nGrid Test Results (3x3 grid centered at obstacle):")
    print(f"{'Offset':<12} {'Test Point':<30} {'Safety Margin':<15} {'Gradient'}")
    print(f"{'-'*80}")

    for dy in offsets:
        for dx in offsets:
            test_point = np.array([cx + dx, cy + dy])
            safety_margin = obs.ellipsoid_safety_margin(test_point, elp_center)
            derivatives = obs.ellipsoid_safety_margin_derivatives(test_point, elp_center)

            point_str = f"[{test_point[0]:.2f}, {test_point[1]:.2f}]"
            offset_str = f"[{dx:+.1f}, {dy:+.1f}]"
            grad_str = f"[{derivatives[0]:.6f}, {derivatives[1]:.6f}]"

            print(f"{offset_str:<12} {point_str:<30} {safety_margin:<15.10f} {grad_str}")

    return True

def main():
    print("="*70)
    print("Comprehensive Obstacle Verification Test - 15 Test Cases")
    print("="*70)

    # Test Case 1: Basic Initialization - Point at Origin
    obs_state1 = np.array([10.0, 5.0, 2.0, 0.0])  # [x, y, v, yaw]
    obs_attr1 = np.array([2.0, 4.5, 1.5])  # [width, length, d_safe]
    elp_center1 = np.array([10.0, 5.0])
    test_point1 = np.array([0.0, 0.0])
    results1 = run_test_case("01: Basic Initialization - Point at Origin",
                          obs_state1, obs_attr1, test_point1, elp_center1)

    # Test Case 2: Safety Margin (Point Near Obstacle, No Rotation)
    obs_state2 = np.array([10.0, 5.0, 2.0, 0.0])
    obs_attr2 = np.array([2.0, 4.5, 1.5])
    elp_center2 = np.array([10.0, 5.0])
    test_point2 = np.array([8.0, 4.0])
    results2 = run_test_case("02: Point Near Obstacle (No Rotation)",
                          obs_state2, obs_attr2, test_point2, elp_center2)

    # Test Case 3: Safety Margin at Boundary
    obs_state3 = np.array([10.0, 5.0, 2.0, 0.0])
    obs_attr3 = np.array([2.0, 4.5, 1.5])
    elp_center3 = np.array([10.0, 5.0])
    obs_temp = obstacle(obs_state3, obs_attr3)
    test_point3 = np.array([10.0 - obs_temp.a, 5.0])  # Exactly at x = center_x - a
    results3 = run_test_case("03: Point at Boundary (No Rotation)",
                          obs_state3, obs_attr3, test_point3, elp_center3)

    # Test Case 4: Safety Margin (With Rotation)
    obs_state4 = np.array([10.0, 5.0, 2.0, np.pi/4])  # 45 degrees rotation
    obs_attr4 = np.array([2.0, 4.5, 1.5])
    elp_center4 = np.array([10.0, 5.0])
    test_point4 = np.array([8.0, 4.0])
    results4 = run_test_case("04: Point Near Obstacle (45° Rotation)",
                          obs_state4, obs_attr4, test_point4, elp_center4)

    # Test Case 5: Derivatives with Rotation
    obs_state5 = np.array([10.0, 5.0, 2.0, np.pi/6])  # 30 degrees rotation
    obs_attr5 = np.array([2.0, 4.5, 1.5])
    elp_center5 = np.array([10.0, 5.0])
    test_point5 = np.array([7.5, 4.5])
    results5 = run_test_case("05: Derivatives (30° Rotation)",
                          obs_state5, obs_attr5, test_point5, elp_center5)

    # Test Case 6: Prediction Trajectory
    obs_state6 = np.array([10.0, 5.0, 2.0, 0.5])
    obs_attr6 = np.array([2.0, 4.5, 1.5])
    elp_center6 = np.array([10.0, 5.0])
    results6 = run_test_case("06: Prediction Trajectory (Moving Obstacle)",
                          obs_state6, obs_attr6, None, None)

    # Test Case 7: Point Far From Obstacle
    obs_state7 = np.array([10.0, 5.0, 2.0, 0.0])
    obs_attr7 = np.array([2.0, 4.5, 1.5])
    elp_center7 = np.array([10.0, 5.0])
    test_point7 = np.array([0.0, 0.0])
    results7 = run_test_case("07: Point Far From Obstacle",
                          obs_state7, obs_attr7, test_point7, elp_center7)

    # Test Case 8: Point Inside Obstacle (Collision)
    obs_state8 = np.array([10.0, 5.0, 2.0, 0.0])
    obs_attr8 = np.array([2.0, 4.5, 1.5])
    elp_center8 = np.array([10.0, 5.0])
    test_point8 = np.array([10.0, 5.0])  # At the center
    results8 = run_test_case("08: Point at Obstacle Center (Collision)",
                          obs_state8, obs_attr8, test_point8, elp_center8)

    # Test Case 9: Very Small Obstacle
    obs_state9 = np.array([10.0, 5.0, 2.0, 0.0])
    obs_attr9 = np.array([0.5, 1.0, 0.1])  # [width, length, d_safe]
    elp_center9 = np.array([10.0, 5.0])
    test_point9 = np.array([9.5, 4.5])
    results9 = run_test_case("09: Edge Case - Very Small Obstacle",
                          obs_state9, obs_attr9, test_point9, elp_center9)

    # Test Case 10: Large Safety Buffer
    obs_state10 = np.array([10.0, 5.0, 2.0, 0.0])
    obs_attr10 = np.array([2.0, 4.5, 5.0])  # Large d_safe
    elp_center10 = np.array([10.0, 5.0])
    test_point10 = np.array([5.0, 2.0])
    results10 = run_test_case("10: Edge Case - Large Safety Buffer",
                           obs_state10, obs_attr10, test_point10, elp_center10)

    # Test Case 11: 90 Degree Rotation
    obs_state11 = np.array([10.0, 5.0, 2.0, np.pi/2])  # 90 degrees
    obs_attr11 = np.array([2.0, 4.5, 1.5])
    elp_center11 = np.array([10.0, 5.0])
    test_point11 = np.array([10.0, 3.0])
    results11 = run_test_case("11: Edge Case - 90° Rotation",
                           obs_state11, obs_attr11, test_point11, elp_center11)

    # Test Case 12: 180 Degree Rotation
    obs_state12 = np.array([10.0, 5.0, 2.0, np.pi])  # 180 degrees
    obs_attr12 = np.array([2.0, 4.5, 1.5])
    elp_center12 = np.array([10.0, 5.0])
    test_point12 = np.array([8.0, 5.0])
    results12 = run_test_case("12: Edge Case - 180° Rotation",
                           obs_state12, obs_attr12, test_point12, elp_center12)

    # Test Case 13: Negative Yaw
    obs_state13 = np.array([10.0, 5.0, 2.0, -np.pi/4])  # -45 degrees
    obs_attr13 = np.array([2.0, 4.5, 1.5])
    elp_center13 = np.array([10.0, 5.0])
    test_point13 = np.array([8.0, 6.0])
    results13 = run_test_case("13: Edge Case - Negative Yaw (-45°)",
                           obs_state13, obs_attr13, test_point13, elp_center13)

    # Test Case 14: Multiple Points Grid
    obs_state14 = np.array([10.0, 5.0, 2.0, 0.0])
    obs_attr14 = np.array([2.0, 4.5, 1.5])
    elp_center14 = np.array([10.0, 5.0])
    results14 = run_grid_test_case("14: Edge Case - 3x3 Point Grid",
                                obs_state14, obs_attr14, elp_center14)

    # Test Case 15: Zero Velocity Obstacle
    obs_state15 = np.array([10.0, 5.0, 0.0, 0.0])  # v = 0
    obs_attr15 = np.array([2.0, 4.5, 1.5])
    elp_center15 = np.array([10.0, 5.0])
    test_point15 = np.array([8.0, 4.0])
    results15 = run_test_case("15: Edge Case - Zero Velocity Obstacle",
                           obs_state15, obs_attr15, test_point15, elp_center15)

    print("\n" + "="*70)
    print("All 15 tests completed!")
    print("="*70)

if __name__ == "__main__":
    main()
