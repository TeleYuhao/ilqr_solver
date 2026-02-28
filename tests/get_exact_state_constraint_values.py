#!/usr/bin/env python3
"""
Compute exact reference values for state_constraint verification tests.
"""

import numpy as np
import sys
sys.path.append('/home/yuhao/Code/te/scripts')

from StateConstraint import StateConstraint
from kinematic_model import KinematicModel
from obstacle_base import obstacle

print("=" * 60)
print("Exact Python Reference Values for StateConstraint")
print("=" * 60)

# Create model
model = KinematicModel()

# Create obstacles
obs_state = np.array([15.0, 2.0, 0.0, 0.0], dtype=np.float64)
obs_attr = np.array([2.0, 4.5, 1.5], dtype=np.float64)
obs = obstacle(obs_state, obs_attr)
obstacle_list = [obs]

# Create StateConstraint
state_dim = 4
control_dim = 2
state_constraint = StateConstraint(state_dim, control_dim, model, obstacle_list)

# Test inputs
step = 10
state = np.array([10.0, 0.0, 7.0, 0.0], dtype=np.float64)  # [x, y, v, yaw]
control = np.array([0.5, 0.0], dtype=np.float64)  # [acceleration, steering]

print(f"step = {step}")
print(f"state = {state}")
print(f"control = {control}")

# Test value
print("\n=== Test 1: value() ===")
val = state_constraint.value(step, state, control)
print(f"cost value = {val:.15f}")

# Compute individual components
velo_up = state_constraint.get_bound_constr(state[2], 10.0, "upper")
velo_low = state_constraint.get_bound_constr(state[2], 0.0, "lower")
print(f"Velocity constraints: up={velo_up:.15f}, low={velo_low:.15f}")

print("\n=== Test 2: gradient_lx() ===")
lx = state_constraint.gradient_lx(step, state, control)
print(f"gradient_lx = {lx}")
for i in range(4):
    print(f"  {lx[i]:.15f}")

print("\n=== Test 3: gradient_lu() ===")
lu = state_constraint.gradient_lu(step, state, control)
print(f"gradient_lu = {lu}")
for i in range(2):
    print(f"  {lu[i]:.15f}")

print("\n=== Test 4: hessian_lxx() ===")
lxx = state_constraint.hessian_lxx(step, state, control)
print(f"hessian_lxx =")
for i in range(4):
    for j in range(4):
        print(f"  {lxx[i,j]:.15f}", end="")
    print()

print("\n=== Test 5: hessian_luu() ===")
luu = state_constraint.hessian_luu(step, state, control)
print(f"hessian_luu =")
for i in range(2):
    for j in range(2):
        print(f"  {luu[i,j]:.15f}", end="")
    print()

print("\n=== Test 6: hessian_lxu() ===")
lxu = state_constraint.hessian_lxu(step, state, control)
print(f"hessian_lxu =")
for i in range(4):
    for j in range(2):
        print(f"  {lxu[i,j]:.15f}", end="")
    print()
