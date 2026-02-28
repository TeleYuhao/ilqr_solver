#!/usr/bin/env python3
"""
Compute exact reference values for kinematic_model verification tests.
"""

import numpy as np
import sys
sys.path.append('/home/yuhao/Code/te/scripts')

from kinematic_model import KinematicModel

print("=" * 60)
print("Exact Python Reference Values for KinematicModel")
print("=" * 60)

model = KinematicModel()

# Test 1 & 2 & 3: gradient_fx, gradient_fu, forward_calculation
print("\n=== Tests 1-3: Common inputs ===")
state = np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float64)
control = np.array([1.0, 0.1], dtype=np.float64)
step = 0.1
print(f"state = {state}")
print(f"control = {control}")
print(f"step = {step}")

print("\n=== Test 1: gradient_fx() ===")
A = model.gradient_fx(state, control, step)
print("A matrix (4x4):")
for i in range(4):
    for j in range(4):
        print(f"  {A[i,j]:.15f}", end="")
    print()

print("\n=== Test 2: gradient_fu() ===")
B = model.gradient_fu(state, control, step)
print("B matrix (4x2):")
for i in range(4):
    for j in range(2):
        print(f"  {B[i,j]:.15f}", end="")
    print()

print("\n=== Test 3: forward_calculation() ===")
next_state = model.forward_calculation(state, control, step)
print(f"next_state = {next_state}")
for i in range(4):
    print(f"  {next_state[i]:.15f}")

# Test 4 & 5: get_vehicle_front_and_rear_centers and derivatives
print("\n=== Tests 4-5: Common inputs ===")
pos = np.array([10.0, 20.0], dtype=np.float64)
yaw = 0.5
print(f"pos = {pos}")
print(f"yaw = {yaw}")

print("\n=== Test 4: get_vehicle_front_and_rear_centers() ===")
front_pnt, rear_pnt = model.get_vehicle_front_and_rear_centers(pos, yaw)
print(f"front_pnt = {front_pnt}")
print(f"  {front_pnt[0]:.15f}")
print(f"  {front_pnt[1]:.15f}")
print(f"rear_pnt = {rear_pnt}")
print(f"  {rear_pnt[0]:.15f}")
print(f"  {rear_pnt[1]:.15f}")

print("\n=== Test 5: get_vehicle_front_and_rear_center_derivatives() ===")
front_deriv, rear_deriv = model.get_vehicle_front_and_rear_center_derivatives(yaw)
print("front_deriv (2x4):")
for i in range(2):
    for j in range(4):
        print(f"  {front_deriv[i,j]:.15f}", end="")
    print()
print("rear_deriv (2x4):")
for i in range(2):
    for j in range(4):
        print(f"  {rear_deriv[i,j]:.15f}", end="")
    print()

# Test 6: init_traj
print("\n=== Test 6: init_traj() ===")
init_state = np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float64)
controls = np.zeros((60, 2), dtype=np.float64)
horizon = 60
print(f"init_state = {init_state}")
print(f"controls = zeros((60, 2))")
print(f"horizon = {horizon}")

states = model.init_traj(init_state, controls, horizon)
print(f"\nstate[0] = {states[0]}")
for i in range(4):
    print(f"  {states[0,i]:.15f}")
print(f"\nstate[1] = {states[1]}")
for i in range(4):
    print(f"  {states[1,i]:.15f}")
print(f"\nstate[60] = {states[60]}")
for i in range(4):
    print(f"  {states[60,i]:.15f}")
