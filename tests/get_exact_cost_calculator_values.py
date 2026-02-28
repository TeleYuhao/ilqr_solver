#!/usr/bin/env python3
"""
Compute exact reference values for cost_calculator verification tests.
"""

import numpy as np
import sys
sys.path.append('/home/yuhao/Code/te/scripts')

from StateConstraint import StateConstraint
from ControlConstraint import ControlConstraint
from state_cost import StateCost
from CostCalculator import CostCalculator
from kinematic_model import KinematicModel
from obstacle_base import obstacle

print("=" * 60)
print("Exact Python Reference Values for CostCalculator")
print("=" * 60)

# Create model
model = KinematicModel()

# Create obstacles
obs_state = np.array([15.0, 2.0, 0.0, 0.0], dtype=np.float64)
obs_attr = np.array([2.0, 4.5, 1.5], dtype=np.float64)
obs = obstacle(obs_state, obs_attr)
obstacle_list = [obs]

# Create reference waypoints (simple straight line)
horizon = 60
num_waypoints = 1000
longit_ref = np.linspace(0, 50, num_waypoints)
lateral_ref = np.linspace(0, 0, num_waypoints)
ref_waypoints = np.vstack((longit_ref, lateral_ref))  # Shape: (2, num_waypoints)

# Create cost components
state_dim = 4
control_dim = 2
# Q matrix matching test.py: diag([1.0, 1.0, 0.5, 0])
Q = np.diag([1.0, 1.0, 0.5, 0.0]).astype(np.float64)
R = np.eye(2, dtype=np.float64)
state_cost = StateCost(Q, R, ref_waypoints, state_dim, control_dim)

state_constraint = StateConstraint(state_dim, control_dim, model, obstacle_list)
control_constraint = ControlConstraint(state_dim, control_dim)

# Create CostCalculator
cost_calculator = CostCalculator(
    state_cost,
    state_constraint,
    control_constraint,
    horizon,
    state_dim,
    control_dim
)

# Create test trajectory
print(f"horizon = {horizon}")

# Initialize states and controls
x0 = np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float64)  # [x, y, v, yaw]
init_u = np.zeros((horizon, 2), dtype=np.float64)

# Forward propagate to get initial trajectory
states = np.zeros((horizon + 1, 4), dtype=np.float64)
states[0] = x0

for i in range(horizon):
    states[i + 1] = model.forward_calculation(states[i], init_u[i])

controls = init_u.copy()

print(f"Initial state x0 = {x0}")
print(f"Number of states: {states.shape[0]}")
print(f"Number of controls: {controls.shape[0]}")

# Test 1: CalculateTotalCost
print("\n=== Test 1: CalculateTotalCost() ===")
total_cost = cost_calculator.CalculateTotalCost(states, controls)
print(f"total_cost = {total_cost:.15f}")

# Compute individual costs for debugging
print("\nIndividual cost components at each timestep:")
cost_calculator.StateCost.get_ref_states(states[:, :2])
state_cost_total = cost_calculator.StateCost.value(0, states[0], np.zeros(2))
constraint_cost_total = 0

print(f"t=0: state_cost = {state_cost_total:.6f}")

for i in range(1, horizon + 1):
    sc = cost_calculator.StateCost.value(i, states[i], controls[i-1])
    stc = cost_calculator.StateConstrints.value(i, states[i], controls[i-1])
    cc = cost_calculator.ControlConstraints.value(i, states[i], controls[i-1])
    state_cost_total += sc
    constraint_cost_total += (stc + cc)
    if i <= 3 or i > horizon - 2:
        print(f"t={i}: state_cost={sc:.6f}, state_constr={stc:.6f}, control_constr={cc:.6f}")

print(f"\nTotal state_cost = {state_cost_total:.15f}")
print(f"Total constraint_cost = {constraint_cost_total:.15f}")
print(f"Sum = {state_cost_total + constraint_cost_total:.15f}")

# Test 2: CalculateDerivates
print("\n=== Test 2: CalculateDerivates() ===")
lx, lxx, lu, luu, lxu = cost_calculator.CalculateDerivates(states, controls)

print(f"lx shape: {lx.shape}")
print(f"lxx shape: {lxx.shape}")
print(f"lu shape: {lu.shape}")
print(f"luu shape: {luu.shape}")
print(f"lxu shape: {lxu.shape}")

# Print some sample values
print("\nSample lx values (first 3 timesteps):")
for i in range(min(3, horizon + 1)):
    print(f"  t={i}: {lx[i]}")

print("\nSample lu values (first 3 timesteps):")
for i in range(min(3, horizon)):
    print(f"  t={i}: {lu[i]}")

print("\nFull lx values (all timesteps):")
for i in range(horizon + 1):
    print(f"  lx[{i}] = [{lx[i][0]:.15f}, {lx[i][1]:.15f}, {lx[i][2]:.15f}, {lx[i][3]:.15f}]")

print("\nFull lu values (all timesteps):")
for i in range(horizon):
    print(f"  lu[{i}] = [{lu[i][0]:.15f}, {lu[i][1]:.15f}]")

print("\nFull lxx diagonal values (all timesteps):")
for i in range(horizon + 1):
    print(f"  lxx[{i}] diag = [{lxx[i][0,0]:.15f}, {lxx[i][1,1]:.15f}, {lxx[i][2,2]:.15f}, {lxx[i][3,3]:.15f}]")

print("\nFull luu diagonal values (all timesteps):")
for i in range(horizon):
    print(f"  luu[{i}] diag = [{luu[i][0,0]:.15f}, {luu[i][1,1]:.15f}]")
