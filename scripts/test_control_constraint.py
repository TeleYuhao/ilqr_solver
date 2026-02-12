"""
ControlConstraint Verification Test
Tests the ControlConstraint implementation against known values for comparison with C++.
"""

import numpy as np
import sys
sys.path.append("/home/yuhao/Code/Panda_ilqr/scripts")

from cost_base import CostFunc
from ControlConstraint import ControlConstraint

def print_array(name, arr, precision=10):
    """Print array with specified precision"""
    if arr.ndim == 1:
        print(f"{name}: [{', '.join(f'{x:.{precision}f}' for x in arr)}]")
    else:
        print(f"{name}:")
        for row in arr:
            print(f"  [{', '.join(f'{x:.{precision}f}' for x in row)}]")

def run_test_case(name, state, control, constraint):
    """Run a single test case and print results"""
    print(f"\n{'='*60}")
    print(f"Test Case: {name}")
    print(f"{'='*60}")
    print(f"State: {state}")
    print(f"Control: {control}")

    # Test value()
    value = constraint.value(0, state, control)
    print(f"\nvalue(): {value:.10f}")

    # Test gradient_lx()
    lx = constraint.gradient_lx(0, state, control)
    print_array("gradient_lx()", lx)

    # Test gradient_lu()
    lu = constraint.gradient_lu(0, state, control)
    print_array("gradient_lu()", lu)

    # Test hessian_lxx()
    lxx = constraint.hessian_lxx(0, state, control)
    print_array("hessian_lxx()", lxx)

    # Test hessian_luu()
    luu = constraint.hessian_luu(0, state, control)
    print_array("hessian_luu()", luu)

    # Test hessian_lxu()
    lxu = constraint.hessian_lxu(0, state, control)
    print_array("hessian_lxu()", lxu)

    return {
        'value': value,
        'lx': lx,
        'lu': lu,
        'lxx': lxx,
        'luu': luu,
        'lxu': lxu
    }

def main():
    print("="*60)
    print("ControlConstraint Verification Test")
    print("="*60)

    # Create ControlConstraint
    constraint = ControlConstraint(4, 2)

    # Test Case 1: Acceleration at Upper Bound
    state1 = np.array([0.0, 0.0, 5.0, 0.0])  # [x, y, v, yaw]
    control1 = np.array([2.0, 0.0])  # [a, delta] - a = A_MAX
    results1 = run_test_case("Acceleration at Upper Bound (a=A_MAX)", state1, control1, constraint)

    # Test Case 2: Acceleration Above Upper Bound
    state2 = np.array([0.0, 0.0, 5.0, 0.0])
    control2 = np.array([2.5, 0.0])  # a > A_MAX (violation)
    results2 = run_test_case("Acceleration Violation (a > A_MAX)", state2, control2, constraint)

    # Test Case 3: Acceleration Below Lower Bound
    state3 = np.array([0.0, 0.0, 5.0, 0.0])
    control3 = np.array([-2.5, 0.0])  # a < A_MIN (violation)
    results3 = run_test_case("Acceleration Violation (a < A_MIN)", state3, control3, constraint)

    # Test Case 4: Steering at Upper Bound
    state4 = np.array([0.0, 0.0, 5.0, 0.0])
    control4 = np.array([0.0, 1.57])  # delta = DELTA_MAX
    results4 = run_test_case("Steering at Upper Bound (delta=DELTA_MAX)", state4, control4, constraint)

    # Test Case 5: Steering Violation (Above Upper Bound)
    state5 = np.array([0.0, 0.0, 5.0, 0.0])
    control5 = np.array([0.0, 2.0])  # delta > DELTA_MAX (violation)
    results5 = run_test_case("Steering Violation (delta > DELTA_MAX)", state5, control5, constraint)

    # Test Case 6: Steering at Lower Bound
    state6 = np.array([0.0, 0.0, 5.0, 0.0])
    control6 = np.array([0.0, -1.57])  # delta = DELTA_MIN
    results6 = run_test_case("Steering at Lower Bound (delta=DELTA_MIN)", state6, control6, constraint)

    # Test Case 7: Both Violations
    state7 = np.array([0.0, 0.0, 5.0, 0.0])
    control7 = np.array([2.5, 2.0])  # Both acceleration and steering violations
    results7 = run_test_case("Both Violations", state7, control7, constraint)

    # Test Case 8: All Constraints Satisfied
    state8 = np.array([0.0, 0.0, 5.0, 0.0])
    control8 = np.array([0.5, 0.3])  # Within bounds
    results8 = run_test_case("All Constraints Satisfied", state8, control8, constraint)

    # Test Case 9: Acceleration at Lower Bound
    state9 = np.array([0.0, 0.0, 5.0, 0.0])
    control9 = np.array([-2.0, 0.0])  # a = A_MIN
    results9 = run_test_case("Acceleration at Lower Bound (a=A_MIN)", state9, control9, constraint)

    # Test Case 10: Mixed - Acceleration violation, steering satisfied
    state10 = np.array([0.0, 0.0, 5.0, 0.0])
    control10 = np.array([2.5, 0.3])  # a > A_MAX, delta within bounds
    results10 = run_test_case("Acceleration Violation Only", state10, control10, constraint)

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
        'test7': results7,
        'test8': results8,
        'test9': results9,
        'test10': results10
    }

if __name__ == "__main__":
    results = main()
