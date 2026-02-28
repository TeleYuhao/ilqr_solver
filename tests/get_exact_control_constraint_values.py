#!/usr/bin/env python3
"""
Compute exact reference values for control_constraint verification tests.
"""

import numpy as np
import sys
sys.path.append('/home/yuhao/Code/te/scripts')

from ControlConstraint import ControlConstraint
from cost_base import CostFunc

print("=" * 60)
print("Exact Python Reference Values for ControlConstraint")
print("=" * 60)

# Create ControlConstraint
state_dim = 4
control_dim = 2

control_constraint = ControlConstraint(state_dim, control_dim)

# Test inputs
step = 0
state = np.array([10.0, 20.0, 5.0, 0.5], dtype=np.float64)  # [x, y, v, yaw]
control = np.array([1.5, 0.8], dtype=np.float64)  # [acceleration, steering]

print(f"step = {step}")
print(f"state = {state}")
print(f"control = {control}")

# Test value
print("\n=== Test 1: value() ===")
val = control_constraint.value(step, state, control)
print(f"cost value = {val:.15f}")

# Compute individual components for debugging
acc_up = control_constraint.get_bound_constr(control[0], 2.0, "upper")
acc_low = control_constraint.get_bound_constr(control[0], -2.0, "lower")
delta_up = control_constraint.get_bound_constr(control[1], 1.57, "upper")
delta_low = control_constraint.get_bound_constr(control[1], -1.57, "lower")
print(f"Constraints: acc_up={acc_up:.15f}, acc_low={acc_low:.15f}")
print(f"             delta_up={delta_up:.15f}, delta_low={delta_low:.15f}")

print("\n=== Test 2: gradient_lx() ===")
lx = control_constraint.gradient_lx(step, state, control)
print(f"gradient_lx = {lx}")
for i in range(4):
    print(f"  {lx[i]:.15f}")

print("\n=== Test 3: gradient_lu() ===")
lu = control_constraint.gradient_lu(step, state, control)
print(f"gradient_lu = {lu}")
for i in range(2):
    print(f"  {lu[i]:.15f}")

print("\n=== Test 4: hessian_lxx() ===")
lxx = control_constraint.hessian_lxx(step, state, control)
print(f"hessian_lxx =")
for i in range(4):
    for j in range(4):
        print(f"  {lxx[i,j]:.15f}", end="")
    print()

print("\n=== Test 5: hessian_luu() ===")
luu = control_constraint.hessian_luu(step, state, control)
print(f"hessian_luu =")
for i in range(2):
    for j in range(2):
        print(f"  {luu[i,j]:.15f}", end="")
    print()

print("\n=== Test 6: hessian_lxu() ===")
lxu = control_constraint.hessian_lxu(step, state, control)
print(f"hessian_lxu =")
for i in range(4):
    for j in range(2):
        print(f"  {lxu[i,j]:.15f}", end="")
    print()
