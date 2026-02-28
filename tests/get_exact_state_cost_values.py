#!/usr/bin/env python3
"""
Compute exact reference values for state_cost verification tests.
"""

import numpy as np
import sys
sys.path.append('/home/yuhao/Code/te/scripts')

from state_cost import StateCost

print("=" * 60)
print("Exact Python Reference Values for StateCost")
print("=" * 60)

# Create StateCost
state_dim = 4
control_dim = 2

Q = np.diag([1.0, 1.0, 0.5, 0]).astype(np.float64)
R = np.diag([1.0, 1.0]).astype(np.float64)

# Reference waypoints (2 x N matrix)
ref_waypoints = np.zeros((2, 10), dtype=np.float64)
for i in range(10):
    ref_waypoints[0, i] = float(i * 5)  # longitudinal
    ref_waypoints[1, i] = 0.0  # lateral

print(f"Q matrix = {Q}")
print(f"R matrix = {R}")
print(f"ref_waypoints shape = {ref_waypoints.shape}")

state_cost = StateCost(Q, R, ref_waypoints, state_dim, control_dim)

# Prepare positions for get_ref_states
# Python expects (N, 2) format for positions, where N should match horizon+1
horizon = state_cost.horizon
positions = np.zeros((horizon + 1, 2), dtype=np.float64)
for i in range(horizon + 1):
    positions[i, 0] = float(i * 5 + 0.5)  # slightly offset from waypoints
    positions[i, 1] = 0.1

print(f"\n=== Test 0: get_ref_states() ===")
print(f"positions shape = {positions.shape}")
print(f"horizon = {horizon}")
state_cost.get_ref_states(positions)

print(f"ref_states shape = {state_cost.ref_states.shape}")
print("Sample ref_states[0] =", state_cost.ref_states[0])
print("Sample ref_states[5] =", state_cost.ref_states[5])

# Test value, gradient, hessian
print("\n=== Tests 1-6: Common inputs ===")
step = 5
state = np.array([25.5, 0.1, 6.0, 0.0], dtype=np.float64)  # [x, y, v, yaw]
control = np.array([0.5, -0.2], dtype=np.float64)  # [acceleration, steering]

print(f"step = {step}")
print(f"state = {state}")
print(f"control = {control}")

print("\n=== Test 1: value() ===")
val = state_cost.value(step, state, control)
print(f"cost value = {val:.15f}")

print("\n=== Test 2: gradient_lx() ===")
lx = state_cost.gradient_lx(step, state, control)
print(f"gradient_lx = {lx}")
for i in range(4):
    print(f"  {lx[i]:.15f}")

print("\n=== Test 3: gradient_lu() ===")
lu = state_cost.gradient_lu(step, state, control)
print(f"gradient_lu = {lu}")
for i in range(2):
    print(f"  {lu[i]:.15f}")

print("\n=== Test 4: hessian_lxx() ===")
lxx = state_cost.hessian_lxx(step, state, control)
print(f"hessian_lxx =")
for i in range(4):
    for j in range(4):
        print(f"  {lxx[i,j]:.15f}", end="")
    print()

print("\n=== Test 5: hessian_luu() ===")
luu = state_cost.hessian_luu(step, state, control)
print(f"hessian_luu =")
for i in range(2):
    for j in range(2):
        print(f"  {luu[i,j]:.15f}", end="")
    print()

print("\n=== Test 6: hessian_lxu() ===")
lxu = state_cost.hessian_lxu(step, state, control)
print(f"hessian_lxu =")
for i in range(4):
    for j in range(2):
        print(f"  {lxu[i,j]:.15f}", end="")
    print()
