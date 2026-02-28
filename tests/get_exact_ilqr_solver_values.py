#!/usr/bin/env python3
"""
Compute exact reference values for ilqr_solver verification tests.
"""

import numpy as np
import sys
sys.path.append('/home/yuhao/Code/te/scripts')

from StateConstraint import StateConstraint
from ControlConstraint import ControlConstraint
from state_cost import StateCost
from CostCalculator import CostCalculator
from ILQR_Core import ilqr
from kinematic_model import KinematicModel
from obstacle_base import obstacle

print("=" * 60)
print("Exact Python Reference Values for ILQRSolver")
print("=" * 60)

# Create model
model = KinematicModel()

# Create reference waypoints
num_waypoints = 1000
longit_ref = np.linspace(0, 50, num_waypoints)
lateral_ref = np.linspace(0, 0, num_waypoints)
ref_waypoints = np.vstack((longit_ref, lateral_ref))

# Create obstacles
obstacle_list = []

# Create cost components
state_dim = 4
control_dim = 2
Q = np.eye(4, dtype=np.float64)
R = np.eye(2, dtype=np.float64)
state_cost = StateCost(Q, R, ref_waypoints, state_dim, control_dim)

state_constraint = StateConstraint(state_dim, control_dim, model, obstacle_list)
control_constraint = ControlConstraint(state_dim, control_dim)

# Create CostCalculator
horizon = 60
cost_calculator = CostCalculator(
    state_cost,
    state_constraint,
    control_constraint,
    horizon,
    state_dim,
    control_dim
)

# Create ILQR solver
ilqr_solver = ilqr(model, cost_calculator)

# Initial state
x0 = np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float64)

print(f"Initial state x0 = {x0}")
print(f"Horizon = {horizon}")
print(f"Max iterations = {ilqr_solver.max_iter}")
print(f"Initial lambda = {ilqr_solver.init_lamb}")

# Initialize trajectory
init_u = np.zeros((horizon, 2), dtype=np.float64)
init_x = model.init_traj(x0, init_u)
init_J = cost_calculator.CalculateTotalCost(init_x, init_u)

print(f"\nInitial cost = {init_J:.15f}")
print(f"Initial states (first 3):")
for i in range(3):
    print(f"  x[{i}] = [{init_x[i, 0]:.6f}, {init_x[i, 1]:.6f}, {init_x[i, 2]:.6f}, {init_x[i, 3]:.6f}]")

# Test backward_pass
print("\n=== Test: backward_pass() ===")
lamb = 20.0
d, K, delt_V = ilqr_solver.backward_pass(init_u, init_x, lamb)

print(f"Expected cost reduction (delt_V) = {delt_V:.15f}")
print(f"d (feedforward gains) shape: {d.shape}")
print(f"K (feedback gains) shape: {K.shape}")

print(f"\nd values (first 3):")
for i in range(3):
    print(f"  d[{i}] = [{d[i, 0]:.15f}, {d[i, 1]:.15f}]")

print(f"\nK[0] (first feedback gain):")
for i in range(2):
    for j in range(4):
        print(f"  K[0][{i},{j}] = {K[0, i, j]:.15f}")

# Test forward_pass
print("\n=== Test: forward_pass() ===")
alpha = 1.0
new_u, new_x = ilqr_solver.forward_pass(init_u, init_x, d, K, alpha)

print(f"new_u shape: {new_u.shape}")
print(f"new_x shape: {new_x.shape}")

print(f"\nnew_x values (first 3):")
for i in range(3):
    print(f"  new_x[{i}] = [{new_x[i, 0]:.15f}, {new_x[i, 1]:.15f}, {new_x[i, 2]:.15f}, {new_x[i, 3]:.15f}]")

new_J = cost_calculator.CalculateTotalCost(new_x, new_u)
print(f"\nNew cost after forward_pass: {new_J:.15f}")
print(f"Cost reduction: {init_J - new_J:.15f}")

# Test iter
print("\n=== Test: iter() ===")
lamb = 20.0
new_u, new_x, new_J, effective = ilqr_solver.iter(init_u, init_x, init_J, lamb)

print(f"Iteration effective: {effective}")
print(f"New cost: {new_J:.15f}")
print(f"Cost reduction: {init_J - new_J:.15f}")
print(f"Final lambda: {lamb:.15f}")

# Test solve (limited iterations for verification)
print("\n=== Test: solve() (3 iterations) ===")
ilqr_solver_test = ilqr(model, cost_calculator)
ilqr_solver_test.max_iter = 3
u_opt, x_opt = ilqr_solver_test.solve(x0)

print(f"\nFinal cost after 3 iterations:")
final_J = cost_calculator.CalculateTotalCost(x_opt, u_opt)
print(f"  {final_J:.15f}")

print(f"\nFinal states (first 3, last 3):")
for i in [0, 1, 2, 58, 59, 60]:
    print(f"  x[{i}] = [{x_opt[i, 0]:.15f}, {x_opt[i, 1]:.15f}, {x_opt[i, 2]:.15f}, {x_opt[i, 3]:.15f}]")

print(f"\nFinal controls (first 3, last 3):")
for i in [0, 1, 2, 57, 58, 59]:
    print(f"  u[{i}] = [{u_opt[i, 0]:.15f}, {u_opt[i, 1]:.15f}]")

# Print all states and controls for verification
print("\n=== All states for verification ===")
for i in range(len(x_opt)):
    print(f"  x[{i}] = [{x_opt[i, 0]:.15f}, {x_opt[i, 1]:.15f}, {x_opt[i, 2]:.15f}, {x_opt[i, 3]:.15f}]")

print("\n=== All controls for verification ===")
for i in range(len(u_opt)):
    print(f"  u[{i}] = [{u_opt[i, 0]:.15f}, {u_opt[i, 1]:.15f}]")
