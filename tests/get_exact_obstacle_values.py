#!/usr/bin/env python3
"""
Compute exact reference values for obstacle verification tests.
"""

import numpy as np
import sys
sys.path.append('/home/yuhao/Code/te/scripts')

from obstacle_base import obstacle

print("=" * 60)
print("Exact Python Reference Values for Obstacle")
print("=" * 60)

# Test inputs
state = np.array([5.0, -0.2, 3.0, -0.0], dtype=np.float64)  # [x, y, v, yaw]
attr = np.array([2.0, 4.5, 1.5], dtype=np.float64)  # [width, length, safety_buffer]

print(f"\nObstacle state = {state}")
print(f"Obstacle attr = {attr}")

# Create obstacle
obs = obstacle(state, attr)

print(f"\nEllipsoid scales:")
print(f"  a = {obs.a:.15f}")
print(f"  b = {obs.b:.15f}")

# Test ellipsoid_safety_margin
print("\n=== Test 1: ellipsoid_safety_margin() ===")
pnt = np.array([5.5, 0.0], dtype=np.float64)
elp_center = np.array([5.0, -0.2], dtype=np.float64)
print(f"pnt = {pnt}")
print(f"elp_center = {elp_center}")

margin = obs.ellipsoid_safety_margin(pnt, elp_center)
print(f"safety_margin = {margin:.15f}")

# Test ellipsoid_safety_margin_derivatives
print("\n=== Test 2: ellipsoid_safety_margin_derivatives() ===")
deriv = obs.ellipsoid_safety_margin_derivatives(pnt, elp_center)
print(f"derivatives = {deriv}")
print(f"  {deriv[0]:.15f}")
print(f"  {deriv[1]:.15f}")

# Test const_velo_prediction
print("\n=== Test 3: const_velo_prediction() ===")
steps = 5  # Just a few steps for testing
pred_traj = obs.const_velo_prediction(state, steps)
print(f"Prediction trajectory (steps={steps}):")
for i, s in enumerate(pred_traj):
    print(f"  step {i}: {s}")
    print(f"    {s[0]:.15f} {s[1]:.15f} {s[2]:.15f} {s[3]:.15f}")
