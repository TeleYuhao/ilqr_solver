# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **iLQR (Iterative Linear Quadratic Regulator)** trajectory optimization library for autonomous vehicle path planning. It implements a model-predictive control algorithm using Riccati backward recursion with exponential barrier functions for soft constraints.

**Dual Implementation:**
- **Python prototypes** in [`scripts/`](scripts/) - for verification, testing, and visualization (uses NumPy)
- **C++ header-only library** in [`include/`](include/) - production implementation using Eigen templates

## Architecture

### Core iLQR Components

```
ILQRSolver
    ├── KinematicModel: Bicycle model with 4D state [x, y, v, yaw], 2D control [accel, steer]
    │   └── gradient_fx(), gradient_fu(): Dynamics Jacobians
    │
    ├── CostCalculator: Aggregates all cost components
    │   ├── StateCost: Quadratic tracking cost to reference waypoints
    │   ├── StateConstraint: Velocity bounds + obstacle avoidance (exponential barriers)
    │   └── ControlConstraint: Acceleration/steering limits (exponential barriers)
    │
    └── Obstacle: Ellipsoid safety margins with constant-velocity prediction
```

### iLQR Algorithm Flow

1. **Backward Pass**: Compute feedback gains **K** and feedforward gains **d** using Riccati recursion
   - Regularization: `Q_uu_regu = Q_uu + λ * I` (adaptive λ)
   - Gains: `d = -Q_uu_inv @ Q_u`, `K = -Q_uu_inv @ Q_ux`

2. **Forward Pass** (Line Search): Try α values [1.0, 0.5, 0.25, 0.125, 0.0625]
   - Accept if cost decreases

3. **Adaptive Regularization**:
   - λ → λ * 0.7 on success
   - λ → λ * 2.0 on failure

### Type System

**C++ Template Pattern:**
```cpp
ILQR_PROBLEM_VARIABLES(T, M, N)  // Defines: State, Control, MatrixLXX, etc.
```

**Python Equivalent:**
```python
self.M, self.N  # state_dim, control_dim
self.State = NDArray[T]  # shape: (M,)
self.MatrixLXX = NDArray[T]  # shape: (M, M)
```

### Exponential Barrier Functions

Soft constraints using: `b(q1, q2, c) = q1 * exp(q2 * c)`

- **Velocity constraints**: upper bound (v - V_MAX), lower bound (V_MIN - v)
- **Obstacle avoidance**: ellipsoid safety margin with vehicle front/rear points
- **Control constraints**: acceleration and steering limits

## Configuration

**Vehicle Parameters** (from [`test.py`](scripts/test.py)):
- Wheelbase (WB): 3.6m, Width: 2.0m, Length: 4.5m, Safety Buffer: 1.5m
- State bounds: velocity [0, 10] m/s
- Control bounds: acceleration [-2, 2] m/s², steering [-1.57, 1.57] rad

**Solver Parameters:**
- Horizon: 60 steps at dt=0.1s (6 second planning)
- Max iterations: 50, λ_init: 20.0, tolerance: 0.001
- Barrier parameters: q1=5.5, q2=5.75

## Running Tests

**Python tests** (from `scripts/` directory):
```bash
cd scripts
python test.py                           # Main demo with visualization
python test_ilqr_comprehensive.py        # Full solver test (no vis)
python test_obstacle_comprehensive.py    # Obstacle tests
```

**C++ tests** (when tests/ directory exists):
```bash
cd tests/build
cmake ..
make
./run_all_tests                           # Run all tests
./run_all_tests --gtest_filter="TestName.*"  # Run specific test
```

## Python/C++ Equivalence

Python files in [`scripts/`](scripts/) mirror C++ headers in [`include/`](include/):

| Python | C++ |
|--------|-----|
| `model_base.py` | `model_base.hpp`, `kinematic_model.hpp` |
| `cost_base.py` | `cost_base.hpp` |
| `state_cost.py` | `state_cost.hpp` |
| `StateConstraint.py` | `state_constraint.hpp` |
| `ControlConstraint.py` | `control_constraint.hpp` |
| `CostCalculator.py` | `cost_calculator.hpp` |
| `obstacle_base.py` | `obstacle.hpp` |
| `ILQR_Core.py` | `ilqr_solver.hpp` |

**Key Implementation Notes:**
- NumPy 1D arrays act as row vectors in matrix multiplication (`@` operator)
- Eigen vectors are column vectors; use `.transpose()` for matrix operations
- Exponential barrier derivatives require careful chain rule application

## Development Context

This is part of a ROS2 Humble workspace. The C++ implementation uses:
- Eigen3 for linear algebra
- Template-based design for generic scalar types
- Aligned allocators for Eigen types in std::vector

When modifying algorithms, maintain consistency between Python and C++ implementations. The Python version serves as the reference for correctness verification.
