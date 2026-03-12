# ALM iLQR Trajectory Optimization

A Python implementation of **Augmented Lagrangian Method (ALM)** combined with **Iterative Linear Quadratic Regulator (iLQR)** for constrained trajectory optimization of autonomous vehicles.

## Overview

This solver addresses the constrained trajectory optimization problem:

```
minimize  Σ [ (x - x_ref)ᵀ Q (x - x_ref) + uᵀ R u ]
subject to:
    - Velocity bounds:        v_min ≤ v ≤ v_max
    - Acceleration limits:    a_min ≤ a ≤ a_max
    - Steering rate limits:   ω_min ≤ ω ≤ ω_max
    - Obstacle avoidance:     d(vehicle, obstacle) ≥ safety_margin
```

Using a **two-loop ALM structure**:
- **Outer loop**: Enforces constraints via Lagrange multiplier updates
- **Inner loop**: Performs iLQR optimization with augmented Lagrangian cost

## Demo

### Dynamic MPC Simulation
![Simulation GIF](simulation.gif)

The animation shows the vehicle (blue) navigating around moving obstacles (red) using Model Predictive Control with ALM iLQR trajectory optimization.

### Single-Shot Trajectory Planning
![Vehicle Trajectory](vehicle_trajectory.png)

Visualization of a single-shot trajectory planning result with static obstacles, showing the vehicle's planned path and orientation at each timestep.

## Features

- **Two-loop ALM optimization**: Outer loop enforces constraints via multiplier updates, inner loop performs iLQR optimization
- **Vehicle model**: 5D bicycle kinematic model [x, y, v, phi, yaw] with 2D control [a, omega]
- **Constraint handling**: Velocity bounds, acceleration limits, steering rate limits, and obstacle avoidance
- **Dynamic simulation**: MPC-style replanning with moving obstacles
- **Visualization**: Trajectory plots and animated GIF generation
- **Two ALM formulations**: V2 (port from ALM_ilqr_v2) and V3 (refactored) implementations

### Vehicle Model: 5D Bicycle Kinematics

```
State:  x = [x, y, v, φ, yaw]
Control: u = [a, ω]

Dynamics:
  ẋ = v * cos(yaw)
  ẏ = v * sin(yaw)
  v̇ = a
  φ̇ = ω
  yaẇ = v * tan(φ) / L
```

Where `L` is the wheelbase and `φ` is the steering angle.

## Installation

```bash
cd ALM_ilqr_v3
pip install numpy matplotlib
```

Requirements:
- Python 3.8+
- numpy
- matplotlib

**Note**: This is a self-contained package with no external dependencies.

## Usage

### Single-shot Planning

```python
from alm_ilqr_core import ALMILQRCore
from alm_model import ALMModel  # V3 formulation
# from ALM_Model_v2 import ALMModelV2  # Alternative: V2 formulation

# Configure
config = {
    'wheelbase': 3.6,
    'horizon': 60,
    'dt': 0.1,
    'state_dim': 5,
    'control_dim': 2,
    'Q': np.diag([10, 10, 1, 0.1, 1]),      # State cost weights
    'R': np.diag([0.1, 0.1]),                # Control cost weights
    'v_max': 10.0, 'v_min': 0.0,             # Velocity bounds
    'acc_max': 2.0, 'acc_min': -2.0,         # Acceleration limits
    'omega_max': 1.57, 'omega_min': -1.57,   # Steering rate limits
    'max_alm_iters': 10,
    'max_ilqr_iters': 10,
    'mu_init': 1.0,
    'violation_tol': 1e-7,
}

# Initialize
model = ALMModel(config)  # or ALMModelV2(config)
solver = ALMILQRCore(model, config)

# Solve
x_opt, u_opt, traj_hist, cost_hist = solver.solve(
    init_state, init_controls, ref_path, obstacles
)
```

### Dynamic Simulation with Animation

```bash
python simulation_animation.py
```

This runs a 6-second simulation with MPC replanning and generates `simulation.gif`.

### Planning with visualization

```bash
python planning_main.py
```

## Code Style

This package follows Google Python Style Guide:
- PEP 8 compliant formatting
- Comprehensive docstrings for all modules and public methods
- Type hints for all function parameters and return values
- Organized imports: standard library → third-party → local
- Descriptive variable names (snake_case for variables, PascalCase for classes)

## File Structure

```
ALM_ilqr_v3/
├── __init__.py              # Package initialization
├── model_base.py            # Abstract base class for kinematic models
├── obstacle.py              # Obstacle class with ellipsoidal safety margins
├── alm_model.py             # V3 ALM vehicle model implementation
├── ALM_Model_v2.py          # V2 ALM vehicle model (ported from ALM_ilqr_v2)
├── alm_ilqr_core.py         # ALM iLQR solver implementation
├── planning_main.py         # Single-shot planning demo
├── simulation_animation.py  # Dynamic simulation with animation
├── simulation.gif           # Generated animation demo
├── vehicle_trajectory.png   # Generated trajectory visualization
└── README.md                # This file
```

### Key Components

| File | Description |
|------|-------------|
| `model_base.py` | Abstract base class defining the model interface |
| `obstacle.py` | Ellipsoidal obstacle safety margin calculation |
| `alm_model.py` | V3 ALM formulation (refactored, production-ready) |
| `ALM_Model_v2.py` | V2 ALM formulation (port from ALM_ilqr_v2) |
| `alm_ilqr_core.py` | Two-loop ALM solver with iLQR inner loop |

## Algorithm Overview

### Two-Loop ALM Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    ALM Outer Loop                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  For k = 1, 2, ..., max_alm_iters:                   │  │
│  │    1. Update mu (penalty weight)                      │  │
│  │    2. Run iLQR inner loop → (x*, u*)                  │  │
│  │    3. Compute constraint violation                    │  │
│  │    4. Update lambda (multipliers)                     │  │
│  │    5. Check convergence (violation < tol)             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    iLQR Inner Loop                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  For t = 1, 2, ..., max_ilqr_iters:                  │  │
│  │    1. Backward Pass (Riccati recursion)              │  │
│  │       - Compute Q_x, Q_u, Q_xx, Q_uu, Q_ux           │  │
│  │       - Compute feedback gains K and feedforward d    │  │
│  │    2. Forward Pass (Line search)                      │  │
│  │       - Try α ∈ {1, 0.5, 0.25, 0.125, ...}           │  │
│  │       - Accept if cost decreases                      │  │
│  │    3. Adaptive regularization (λ)                     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Mathematical Formulation

#### Projection-Based ALM (V3 and corrected V2)

The augmented Lagrangian cost uses a **projection operator** P(x) = min(x, 0):

```
L_alm(x, u, λ) = L_tracking(x, u) + Σ_i  ½/μ * (||P(λ_i - μ*c_i)||² - ||λ_i||²)
```

Where:
- `c_i` = constraint value at timestep i
- `λ_i` = Lagrange multiplier at timestep i
- `μ` = penalty parameter
- `P(z)` = projection onto non-positive orthant: `min(z, 0)`

#### Multiplier Update

```
λ_i ← P(λ_i - μ * c_i) = min(λ_i - μ * c_i, 0)
```

#### Derivatives with Projection Jacobian

The projection Jacobian `J_P(z)` is a diagonal matrix:
- `J_P[i,i] = 1` if `z_i ≤ 0`
- `J_P[i,i] = 0` if `z_i > 0`

Cost derivatives include the projection Jacobian:
```
l_u = l_u' - (J_P @ J_c_u)ᵀ @ P(λ - μ*c)
l_uu = l_uu' + (J_P @ J_c_u)ᵀ @ (J_P @ J_c_u) * μ
```

#### Comparison: V2 vs V3 ALM Formulations

Both formulations use the **same projection-based approach**. The differences are minimal:

| Aspect | V2 (ALMModelV2) | V3 (ALMModel) |
|--------|-----------------|---------------|
| **Cost formula** | `½/μ (||P(λ-μc)||² - ||λ||²)` | Same |
| **Multiplier update** | `P(λ - μ*c)` | Same |
| **Projection** | `min(x, 0)` (non-positive) | Same |
| **Derivatives** | With projection Jacobian | Same |
| **Code heritage** | Ported from `ALM_ilqr_v2` | Refactored V3 codebase |

**Note**: The V2 model in this package is the corrected version, properly implementing the projection-based formulation.

### Constraint Handling

The solver handles four types of constraints using **exponential barrier functions**:

#### 1. Control Constraints
```
a ≤ acc_max        →  c = a - acc_max
a ≥ acc_min        →  c = acc_min - a
ω ≤ omega_max      →  c = ω - omega_max
ω ≥ omega_min      →  c = omega_min - ω
```

#### 2. Velocity Constraints
```
v ≤ v_max          →  c = v - v_max
v ≥ v_min          →  c = v_min - v
```

#### 3. Obstacle Avoidance (Ellipsoidal Safety Margin)

For each obstacle, we compute safety margins at **front and rear axles**:

```
c_obs = safety_radius² - [(x - x_obs)²/a² + (y - y_obs)²/b²]
```

Where (a, b) are the obstacle semi-axes and `safety_radius` includes vehicle dimensions.

**Total constraint vector per timestep**: `c = [ctrl, velocity, obstacle_front, obstacle_rear]`

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `horizon` | Planning horizon steps | 60 |
| `dt` | Time step [s] | 0.1 |
| `wheelbase` | Vehicle wheelbase [m] | 3.6 |
| `state_dim` | State dimension | 5 |
| `control_dim` | Control dimension | 2 |
| `Q` | State cost matrix (5×5) | diag([10,10,1,0.1,1]) |
| `R` | Control cost matrix (2×2) | diag([0.1, 0.1]) |
| `v_max`, `v_min` | Velocity bounds [m/s] | 10.0, 0.0 |
| `acc_max`, `acc_min` | Acceleration limits [m/s²] | 2.0, -2.0 |
| `omega_max`, `omega_min` | Steering rate limits [rad/s] | 1.57, -1.57 |
| `max_alm_iters` | Maximum outer loop iterations | 10 |
| `max_ilqr_iters` | Maximum inner loop iterations | 10 |
| `mu_init` | Initial penalty parameter | 1.0 |
| `mu_gain` | Mu increase factor | 8 |
| `violation_tol` | Constraint violation tolerance | 1e-7 |

## Troubleshooting

### Constraint Violation Not Decreasing

If constraint violations remain large after several ALM iterations:

1. **Increase `mu_init`** - Start with a larger penalty (e.g., 10.0)
2. **Increase `mu_gain`** - More aggressive mu scaling (e.g., 16)
3. **Check initial trajectory** - Ensure it's not grossly infeasible
4. **Verify constraint definitions** - Check obstacle positions and bounds

### iLQR Not Converging

If the inner loop fails to converge:

1. **Increase `max_ilqr_iters`** - Allow more iterations (e.g., 50)
2. **Tune cost weights** - Reduce Q/R weights for smoother gradients
3. **Reduce horizon** - Shorter planning may be easier
4. **Check initial guess** - Use a feasible initialization

### Slow Performance

For faster computation:

1. **Reduce horizon** - Try 30-40 steps instead of 60
2. **Reduce iterations** - Lower `max_ilqr_iters` to 10-20
3. **Fewer obstacles** - Limit obstacle count in the scene
4. **Warm-start** - Use previous solution as initial guess

## References

### iLQR Algorithm
- Li, D., & Todorov, E. (2008). "Iterative linear quadratic regulator design for nonlinear biological movement systems." *ICRA*

### Augmented Lagrangian Method
- Hestenes, M. R. (1969). "Multiplier and penalty methods." *Journal of Optimization Theory and Applications*
- Bertsekas, D. P. (1976). "On penalty and multiplier methods for constrained optimization problems." *SIAM Journal on Control*

### Projected Gradient Methods
- Rockafellar, R. T. (1973). "The multiplier method of Hestenes and Powell applied to convex programming." *Journal of Optimization Theory and Applications*

## License

This implementation is provided for research and educational purposes.

## Contributing

This package follows Google Python Style Guide. When contributing:
- Maintain PEP 8 compliance
- Add docstrings for all new functions
- Include type hints for parameters
- Test both V2 and V3 models for consistency
