#!/usr/bin/env python3
"""
Compute exact reference values for ilqr_solver verification tests.
Uses Q = diag([1.0, 1.0, 0.5, 0]) matching test.py
"""

import numpy as np
import sys
sys.path.append('/home/yuhao/Code/te/scripts')

from StateConstraint import StateConstraint
from ControlConstraint import ControlConstraint
from state_cost import StateCost
from CostCalculator import CostCalculator
from kinematic_model import KinematicModel
from ILQR_Core import ilqr

print("=" * 60)
print("Exact Python Reference Values for ILQR Solver")
print("Q = diag([1.0, 1.0, 0.5, 0])")
print("=" * 60)

# Create model
model = KinematicModel()

# Create reference waypoints (simple straight line)
horizon = 60
num_waypoints = 1000
longit_ref = np.linspace(0, 50, num_waypoints)
lateral_ref = np.linspace(0, 0, num_waypoints)
ref_waypoints = np.vstack((longit_ref, lateral_ref))  # Shape: (2, num_waypoints)

# Create cost components (no obstacles for basic test)
state_dim = 4
control_dim = 2
Q = np.diag([1.0, 1.0, 0.5, 0.0]).astype(np.float64)  # Matching test.py
R = np.eye(2, dtype=np.float64)
state_cost = StateCost(Q, R, ref_waypoints, state_dim, control_dim)

# Empty obstacle list
obstacle_list = []
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

# Create ILQR solver
ilqr_solver = ilqr(model, cost_calculator)

# Test 1: solve() - 3 iterations
print("\n=== Test 1: solve() - 3 iterations ===")
x0 = np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float64)

# Set max iterations to 3 (matching C++ test)
ilqr_solver.max_iter = 3
u, x = ilqr_solver.solve(x0)

print(f"Number of states: {len(x)}")
print(f"Number of controls: {len(u)}")

# Calculate final cost
final_cost = cost_calculator.CalculateTotalCost(x, u)
print(f"Final cost: {final_cost:.15f}")

print("\n  Sample states:")
print(f"    x[0]: [{x[0][0]:.15f}, {x[0][1]:.15f}, {x[0][2]:.15f}, {x[0][3]:.15f}]")
print(f"    x[30]: [{x[30][0]:.15f}, {x[30][1]:.15f}, {x[30][2]:.15f}, {x[30][3]:.15f}]")
print(f"    x[60]: [{x[60][0]:.15f}, {x[60][1]:.15f}, {x[60][2]:.15f}, {x[60][3]:.15f}]")

print("\n  Sample controls:")
print(f"    u[0]: [{u[0][0]:.15f}, {u[0][1]:.15f}]")
print(f"    u[30]: [{u[30][0]:.15f}, {u[30][1]:.15f}]")
print(f"    u[59]: [{u[59][0]:.15f}, {u[59][1]:.15f}]")

# Test 2: backward_pass
print("\n=== Test 2: backward_pass() ===")
# Initialize trajectory
init_u = np.zeros((horizon, 2), dtype=np.float64)
init_x = model.init_traj(x0, init_u)

# Get initial cost
init_J = cost_calculator.CalculateTotalCost(init_x, init_u)

# Run backward pass
lambda_reg = 20.0
d, K, delt_V = ilqr_solver.backward_pass(init_u, init_x, lamb=lambda_reg)

print(f"Expected cost reduction (delt_V): {delt_V:.15f}")

print("\n  d[0]:")
print(f"    [0]: {d[0][0]:.15f}")
print(f"    [1]: {d[0][1]:.15f}")

print("\n  d[1]:")
print(f"    [0]: {d[1][0]:.15f}")
print(f"    [1]: {d[1][1]:.15f}")

print("\n  K[0]:")
print(f"    [0,0]: {K[0][0,0]:.15f}")
print(f"    [0,1]: {K[0][0,1]:.15f}")
print(f"    [0,2]: {K[0][0,2]:.15f}")
print(f"    [0,3]: {K[0][0,3]:.15f}")
print(f"    [1,0]: {K[0][1,0]:.15f}")
print(f"    [1,1]: {K[0][1,1]:.15f}")
print(f"    [1,2]: {K[0][1,2]:.15f}")
print(f"    [1,3]: {K[0][1,3]:.15f}")
