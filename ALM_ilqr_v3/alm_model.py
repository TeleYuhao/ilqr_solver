"""ALM vehicle model with augmented Lagrangian constraints.

This module implements a kinematic bicycle model with 5D state [x, y, v, phi, yaw]
and 2D control [a, omega] for trajectory optimization. It extends ModelBase with
Augmented Lagrangian Method (ALM) components for handling state and control
constraints including velocity bounds, acceleration limits, and obstacle avoidance.

The model uses ellipsoidal safety margins for obstacle constraints computed at
front and rear axle positions. ALM multipliers (lambda) and penalty weights (mu)
are updated based on constraint violation.

Example:
    >>> config = {'wheelbase': 3.6, 'dt': 0.1, 'horizon': 60, ...}
    >>> model = ALMModel(config)
    >>> model.init_multipliers(obstacles)
    >>> cost = model.compute_cost_alm(states, controls, ref_path, obstacles)
"""

from typing import Dict, List, Tuple

import numpy as np

from model_base import ModelBase
from obstacle import Obstacle


# Vehicle parameter constants
VEHICLE_WIDTH = 2.0  # meters
VEHICLE_LENGTH = 4.5  # meters
SAFETY_BUFFER = 1.5  # meters


class ALMModel(ModelBase):
    """ALM vehicle model for constrained trajectory optimization.

    Attributes:
        wheelbase: Vehicle wheelbase in meters.
        config: Configuration dictionary containing model parameters.
        mu: ALM penalty parameter.
        i_mu: Diagonal penalty matrices for each timestep.
        lambda_alm: Lagrange multipliers for each constraint at each timestep.
    """

    def __init__(self, config: Dict) -> None:
        """Initialize the ALM vehicle model.

        Args:
            config: Dictionary containing model configuration parameters including
                wheelbase, dt, horizon, state_dim, control_dim, Q, R, ref_velo,
                and constraint bounds (v_max, v_min, acc_max, acc_min, etc.).
        """
        super().__init__(config)
        self.wheelbase = config['wheelbase']
        self.config = config

    def forward_calculation(
        self,
        state: np.ndarray,
        control: np.ndarray
    ) -> np.ndarray:
        """Compute next state using bicycle kinematic model.

        Args:
            state: Current state [x, y, v, phi, yaw].
            control: Control input [acceleration, steering_angle_rate].

        Returns:
            Next state vector.
        """
        x, y, v, phi, yaw = state
        a, omega = control
        dt = self.config['dt']
        next_state = np.array([
            x + v * np.cos(yaw) * dt,
            y + v * np.sin(yaw) * dt,
            v + a * dt,
            phi + omega * dt,
            yaw + v * np.tan(phi) / self.wheelbase * dt
        ])
        return next_state

    def init_traj(
        self,
        init_state: np.ndarray,
        controls: np.ndarray,
        horizon: int = 60
    ) -> np.ndarray:
        """Initialize state trajectory by forward simulation.

        Args:
            init_state: Initial state vector.
            controls: Control sequence of shape (horizon, control_dim).
            horizon: Planning horizon length.

        Returns:
            State trajectory of shape (horizon+1, state_dim).
        """
        states = np.zeros((horizon + 1, self.state_dim))
        states[0] = init_state
        for i in range(1, horizon + 1):
            states[i] = self.forward_calculation(states[i - 1], controls[i - 1])
        return states

    def get_jacobian(
        self,
        state: np.ndarray,
        control: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Jacobian matrices of the dynamics.

        Args:
            state: Current state [x, y, v, phi, yaw].
            control: Control input [a, omega].

        Returns:
            Tuple of (A, B) where:
                - A: State transition Jacobian (df/dx) of shape (5, 5).
                - B: Control Jacobian (df/du) of shape (5, 2).
        """
        x, y, v, phi, yaw = state
        a, omega = control
        dt = self.config['dt']
        lw = self.wheelbase

        a_matrix = np.array([
            [1, 0, dt * np.cos(yaw), 0, -v * dt * np.sin(yaw)],
            [0, 1, dt * np.sin(yaw), 0, v * dt * np.cos(yaw)],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, dt * np.tan(phi) / lw, dt * v / (lw * np.cos(phi)**2), 1]
        ])

        b_matrix = np.array([
            [0, 0],
            [0, 0],
            [dt, 0],
            [0, dt],
            [0, 0]
        ])

        return a_matrix, b_matrix

    def init_multipliers(self, obstacles: List[Obstacle]) -> None:
        """Initialize ALM multipliers and penalty matrices.

        Args:
            obstacles: List of obstacle objects for determining constraint dimension.
        """
        constraint_dim = (2 * self.control_dim + 2 + 2 * len(obstacles))
        self.mu = 1.0
        self.i_mu = np.stack(
            [np.eye(constraint_dim) for _ in range(self.horizon)], axis=0
        )
        self.lambda_alm = np.zeros((self.horizon, constraint_dim))

    def update_lambda(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        obstacles: List[Obstacle]
    ) -> None:
        """Update Lagrange multipliers using projected gradient step.

        Args:
            states: State trajectory.
            controls: Control sequence.
            obstacles: List of obstacle objects.
        """
        for i in range(1, self.horizon + 1):
            constraint = self.compute_constraint(states, controls, obstacles, i)
            self.lambda_alm[i - 1] = self._projection(
                self.lambda_alm[i - 1] - self.mu * constraint
            )

    def update_mu(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        obstacles: List[Obstacle]
    ) -> None:
        """Update penalty matrix I_mu based on active constraints.

        Only constraints with positive multiplier or violation are penalized.

        Args:
            states: State trajectory.
            controls: Control sequence.
            obstacles: List of obstacle objects.
        """
        if not hasattr(self, 'i_mu'):
            self.init_multipliers(obstacles)

        i_mu_next = self.i_mu.copy()
        constraint_dim = 2 * self.control_dim + 2 + 2 * len(obstacles)

        for i in range(1, self.horizon + 1):
            constraint = self.compute_constraint(states, controls, obstacles, i)
            for j in range(constraint_dim):
                if self.lambda_alm[i - 1, j] == 0 and constraint[j] <= 0:
                    i_mu_next[i - 1, j, j] = 0
                else:
                    i_mu_next[i - 1, j, j] = self.mu

        self.i_mu = i_mu_next

    def update_mu_scalar(self, factor: float = 2.0) -> None:
        """Increase penalty parameter mu by multiplicative factor.

        Called when constraint violation is large to increase penalty weight
        and force constraint satisfaction.

        Args:
            factor: Multiplicative factor for increasing mu.
        """
        self.mu *= factor
        for i in range(self.horizon):
            for j in range(self.i_mu.shape[1]):
                if self.i_mu[i, j, j] > 0:
                    self.i_mu[i, j, j] = self.mu

    def compute_violation(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        obstacles: List[Obstacle]
    ) -> float:
        """Compute total constraint violation (positive values only).

        Args:
            states: State trajectory.
            controls: Control sequence.
            obstacles: List of obstacle objects.

        Returns:
            Total constraint violation (sum of positive constraint values).
        """
        violation = 0
        for i in range(1, self.horizon + 1):
            constraint = self.compute_constraint(states, controls, obstacles, i)
            violation += np.sum(np.maximum(0, constraint))
        return float(violation)

    def get_center(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute front and rear axle center positions.

        Args:
            state: Vehicle state [x, y, v, phi, yaw].

        Returns:
            Tuple of (front_center, rear_center) as 2D position arrays.
        """
        pos = state[:2]
        yaw = state[4]
        half_wheelbase_vec = 0.5 * self.wheelbase * np.array([
            np.cos(yaw), np.sin(yaw)
        ])
        front_pnt = pos + half_wheelbase_vec
        rear_pnt = pos - half_wheelbase_vec
        return front_pnt, rear_pnt

    def _get_axle_derivatives(
        self,
        yaw: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute derivatives of front/rear axle positions w.r.t. state.

        Args:
            yaw: Vehicle heading angle in radians.

        Returns:
            Tuple of (front_jacobian, rear_jacobian) of shape (2, 5).
        """
        half_wheelbase = 0.5 * self.wheelbase

        front_jacobian = np.array([
            [1.0, 0.0, 0.0, 0.0, half_wheelbase * (-np.sin(yaw))],
            [0.0, 1.0, 0.0, 0.0, half_wheelbase * np.cos(yaw)]
        ])

        rear_jacobian = np.array([
            [1.0, 0.0, 0.0, 0.0, -half_wheelbase * (-np.sin(yaw))],
            [0.0, 1.0, 0.0, 0.0, -half_wheelbase * np.cos(yaw)]
        ])

        return front_jacobian, rear_jacobian

    def get_reference_state(
        self,
        states: np.ndarray,
        ref_waypoints: np.ndarray
    ) -> np.ndarray:
        """Find reference states for each trajectory point.

        Args:
            states: State trajectory of shape (horizon+1, state_dim).
            ref_waypoints: Reference waypoints of shape (2, n_waypoints).

        Returns:
            Reference states of shape (horizon+1, state_dim).
        """
        pos = states[:, :2]
        ref_waypoints_reshaped = ref_waypoints.transpose()[:, :, np.newaxis]
        distances = np.sum((pos.T - ref_waypoints_reshaped) ** 2, axis=1)
        arg_min_dist_indices = np.argmin(distances, axis=0)
        ref_exact_points = ref_waypoints[:, arg_min_dist_indices]

        ref_states = np.vstack([
            ref_exact_points,
            np.full(self.horizon + 1, self.config['ref_velo']),
            np.zeros(self.horizon + 1),
            np.zeros(self.horizon + 1)
        ]).T
        return ref_states

    def compute_constraint(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        obstacles: List[Obstacle],
        step: int
    ) -> np.ndarray:
        """Compute constraint values at a specific timestep.

        Constraints include:
        - Acceleration upper/lower bounds
        - Steering rate upper/lower bounds
        - Velocity upper/lower bounds
        - Obstacle safety margins (front and rear axles)

        Args:
            states: State trajectory.
            controls: Control sequence.
            obstacles: List of obstacle objects.
            step: Timestep index (1 to horizon).

        Returns:
            Constraint values (positive means violation).
        """
        x, y, v, phi, yaw = states[step]
        a, omega = controls[step - 1]

        acc_up_constraint = self.get_bound_constr(a, self.config['acc_max'], 'upper')
        acc_low_constraint = self.get_bound_constr(a, self.config['acc_min'], 'lower')
        omega_up_constraint = self.get_bound_constr(omega, self.config['omega_max'], 'upper')
        omega_low_constraint = self.get_bound_constr(omega, self.config['omega_min'], 'lower')
        velo_up_constraint = self.get_bound_constr(v, self.config['v_max'], 'upper')
        velo_low_constraint = self.get_bound_constr(v, self.config['v_min'], 'lower')

        constraints = [
            acc_up_constraint, acc_low_constraint,
            omega_up_constraint, omega_low_constraint,
            velo_up_constraint, velo_low_constraint
        ]

        front_center, rear_center = self.get_center(states[step])

        for obs in obstacles:
            constraints.append(
                obs.ellipsoid_safety_margin(
                    front_center, obs.prediction_traj[step][:2]
                )
            )
            constraints.append(
                obs.ellipsoid_safety_margin(
                    rear_center, obs.prediction_traj[step][:2]
                )
            )
        return np.array(constraints)

    def compute_cost_alm(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        ref_path: np.ndarray,
        obstacles: List[Obstacle]
    ) -> float:
        """Compute ALM-augmented cost including constraint penalties.

        Args:
            states: State trajectory.
            controls: Control sequence.
            ref_path: Reference waypoints.
            obstacles: List of obstacle objects.

        Returns:
            Total cost including tracking cost and ALM penalty terms.
        """
        ref_states = self.get_reference_state(states, ref_path)
        self.update_mu(states, controls, obstacles)

        state_cost = np.sum(
            ((states - ref_states) @ self.config['Q']) * (states - ref_states)
        )
        control_cost = np.sum(controls @ self.config['R'] * controls)

        cost = state_cost + control_cost
        for i in range(1, self.horizon + 1):
            constraint = self.compute_constraint(states, controls, obstacles, i)
            lambda_proj = self._projection(
                self.lambda_alm[i - 1] - self.mu * constraint
            )
            cost += (0.5 / self.mu * (
                lambda_proj.T @ lambda_proj -
                self.lambda_alm[i - 1].T @ self.lambda_alm[i - 1]
            ))

        return float(cost)

    def compute_deri(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        obstacles: List[Obstacle],
        step: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute constraint Jacobians with respect to state and control.

        Args:
            states: State trajectory.
            controls: Control sequence.
            obstacles: List of obstacle objects.
            step: Timestep index.

        Returns:
            Tuple of (jacobian_x, jacobian_u) where:
                - jacobian_x: Constraint derivative w.r.t. state.
                - jacobian_u: Constraint derivative w.r.t. control.
        """
        state = states[step]

        acc_up_du = np.array([1, 0])
        acc_low_du = np.array([-1, 0])
        omega_up_du = np.array([0, 1])
        omega_low_du = np.array([0, -1])
        velo_up_dx = np.array([0, 0, 1, 0, 0])
        velo_low_dx = np.array([0, 0, -1, 0, 0])

        front_center, rear_center = self.get_center(state)
        ego_front_over_state, ego_rear_over_state = self._get_axle_derivatives(
            state[4]
        )

        ctrl_du = np.array([
            acc_up_du, acc_low_du, omega_up_du, omega_low_du
        ])
        state_dx = [velo_up_dx, velo_low_dx]

        for obs in obstacles:
            obs_center = obs.prediction_traj[step][:2]
            front_deri = obs.ellipsoid_safety_margin_derivatives(
                front_center, obs_center
            )
            rear_deri = obs.ellipsoid_safety_margin_derivatives(
                rear_center, obs_center
            )
            state_dx.append(front_deri @ ego_front_over_state)
            state_dx.append(rear_deri @ ego_rear_over_state)

        return np.array(state_dx), np.array(ctrl_du)

    def get_derivates_alm(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        ref_path: np.ndarray,
        obstacles: List[Obstacle]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute cost function derivatives for iLQR backward pass.

        Args:
            states: State trajectory.
            controls: Control sequence.
            ref_path: Reference waypoints.
            obstacles: List of obstacle objects.

        Returns:
            Tuple of (l_x, l_u, l_xx, l_uu, l_ux) containing:
                - l_x: Cost gradient w.r.t. state.
                - l_u: Cost gradient w.r.t. control.
                - l_xx: Cost Hessian w.r.t. state.
                - l_uu: Cost Hessian w.r.t. control.
                - l_ux: Cost cross-Hessian.
        """
        ref_states = self.get_reference_state(states, ref_path)
        self.update_mu(states, controls, obstacles)

        lxs = np.zeros((self.horizon + 1, self.state_dim))
        lus = np.zeros((self.horizon, self.control_dim))
        lxxs = np.zeros((self.horizon + 1, self.state_dim, self.state_dim))
        luus = np.zeros((self.horizon, self.control_dim, self.control_dim))
        luxs = np.zeros((self.horizon, self.control_dim, self.state_dim))

        for i in range(1, self.horizon + 1):
            state = states[i]
            control = controls[i - 1]
            constraint = self.compute_constraint(states, controls, obstacles, i)

            control_constraint = constraint[:2 * self.control_dim]
            state_constraint = constraint[2 * self.control_dim:]

            i_mu_ctrl = self.i_mu[i - 1][:2 * self.control_dim, :2 * self.control_dim]
            i_mu_state = self.i_mu[i - 1][2 * self.control_dim:, 2 * self.control_dim:]

            lambda_ctrl = self.lambda_alm[i - 1][:2 * self.control_dim]
            lambda_state = self.lambda_alm[i - 1][2 * self.control_dim:]

            state_error = state - ref_states[i]
            l_u_prime = 2 * control @ self.config['R']
            l_uu_prime = 2 * self.config['R']

            constraint_jx, constraint_ju = self.compute_deri(
                states, controls, obstacles, i
            )

            lambda_proj = self._projection(
                self.lambda_alm[i - 1] - self.mu * constraint
            )
            proj_jac = self._projection_jacobian(
                self.lambda_alm[i - 1] - self.mu * constraint
            )

            lus[i - 1] = l_u_prime - (
                proj_jac[:2 * self.control_dim, :2 * self.control_dim] @
                constraint_ju
            ).T @ lambda_proj[:2 * self.control_dim]

            jac_cu = proj_jac[:2 * self.control_dim, :2 * self.control_dim] @ constraint_ju
            luus[i - 1] = l_uu_prime + jac_cu.T @ jac_cu * self.mu

            lx_prime = 2 * state_error @ self.config['Q']
            lxx_prime = 2 * self.config['Q']

            lxs[i] = lx_prime - (
                proj_jac[2 * self.control_dim:, 2 * self.control_dim:] @
                constraint_jx
            ).T @ lambda_proj[2 * self.control_dim:]

            jac_cx = proj_jac[2 * self.control_dim:, 2 * self.control_dim:] @ constraint_jx
            lxxs[i] = lxx_prime + jac_cx.T @ jac_cx * self.mu

        return lxs, lus, lxxs, luus, luxs

    def _projection(self, x: np.ndarray) -> np.ndarray:
        """Project vector onto non-positive orthant.

        Args:
            x: Input vector.

        Returns:
            Projected vector (max(x, 0) becomes 0).
        """
        return np.minimum(x, 0)

    def _projection_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Compute Jacobian of the projection operator.

        Args:
            x: Input vector.

        Returns:
            Jacobian matrix (diagonal, 1 where x <= 0, 0 where x > 0).
        """
        jac = np.zeros((x.size, x.size))
        for i in range(x.size):
            jac[i, i] = 0 if x[i] > 0 else 1
        return jac

    def _projection_hessian(self, x: np.ndarray) -> np.ndarray:
        """Compute Hessian of the projection operator (zero matrix).

        Args:
            x: Input vector.

        Returns:
            Zero matrix of shape (x.size, x.size).
        """
        return np.zeros((x.size, x.size))

    def compute_cost(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        ref_path: np.ndarray,
        collision_constraint: List
    ) -> float:
        """Compute the cost of a trajectory (non-ALM version).

        Note: This method is provided for compatibility with ModelBase interface.
        For ALM optimization, use compute_cost_alm instead.

        Args:
            states: State trajectory.
            controls: Control sequence.
            ref_path: Reference path waypoints.
            collision_constraint: Collision constraints.

        Returns:
            Total cost value (tracking + control cost, no constraints).
        """
        ref_states = self.get_reference_state(states, ref_path)

        state_cost = np.sum(
            ((states - ref_states) @ self.config['Q']) * (states - ref_states)
        )
        control_cost = np.sum(controls @ self.config['R'] * controls)

        return float(state_cost + control_cost)

    def get_derivates(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        ref_path: np.ndarray,
        collision_constraint: List
    ) -> Tuple:
        """Compute cost derivatives for iLQR backward pass (non-ALM version).

        Note: This method is provided for compatibility with ModelBase interface.
        For ALM optimization, use get_derivates_alm instead.

        Args:
            states: State trajectory.
            controls: Control sequence.
            ref_path: Reference path waypoints.
            collision_constraint: Collision constraints.

        Returns:
            Tuple of (l_x, l_u, l_xx, l_uu, l_ux) derivatives.
        """
        ref_states = self.get_reference_state(states, ref_path)

        lxs = np.zeros((self.horizon + 1, self.state_dim))
        lus = np.zeros((self.horizon, self.control_dim))
        lxxs = np.zeros((self.horizon + 1, self.state_dim, self.state_dim))
        luus = np.zeros((self.horizon, self.control_dim, self.control_dim))
        luxs = np.zeros((self.horizon, self.control_dim, self.state_dim))

        for i in range(1, self.horizon + 1):
            state = states[i]
            control = controls[i - 1]

            state_error = state - ref_states[i]
            l_u_prime = 2 * control @ self.config['R']
            l_uu_prime = 2 * self.config['R']

            lx_prime = 2 * state_error @ self.config['Q']
            lxx_prime = 2 * self.config['Q']

            lus[i - 1] = l_u_prime
            luus[i - 1] = l_uu_prime
            lxs[i] = lx_prime
            lxxs[i] = lxx_prime

        return lxs, lus, lxxs, luus, luxs
