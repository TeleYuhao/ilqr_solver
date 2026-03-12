"""Abstract base class for kinematic models in trajectory optimization.

This module defines the ModelBase abstract class that all vehicle models should
inherit from. It provides the interface for dynamics simulation, cost computation,
and constraint handling for iLQR trajectory optimization.

The base class includes utility methods for:
- Exponential barrier functions for soft constraints
- ALM (Augmented Lagrangian Method) penalty terms
- Bound constraint computation
- Projection operators

Example:
    >>> class MyModel(ModelBase):
    ...     def forward_calculation(self, state, control):
    ...         # Implement dynamics
    ...         return next_state
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


class ModelBase(ABC):
    """Abstract base class for kinematic models.

    Attributes:
        state_dim: Dimension of the state vector.
        control_dim: Dimension of the control vector.
        horizon: Planning horizon length.
    """

    def __init__(self, config: Dict) -> None:
        """Initialize the model base.

        Args:
            config: Configuration dictionary containing state_dim, control_dim,
                and horizon parameters.
        """
        self.state_dim = config['state_dim']
        self.control_dim = config['control_dim']
        self.horizon = config['horizon']

    @abstractmethod
    def forward_calculation(
        self,
        state: np.ndarray,
        control: np.ndarray
    ) -> np.ndarray:
        """Compute next state from current state and control.

        Args:
            state: Current state vector.
            control: Control input vector.

        Returns:
            Next state vector.
        """
        pass

    @abstractmethod
    def get_jacobian(
        self,
        state: np.ndarray,
        control: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute dynamics Jacobians.

        Args:
            state: Current state vector.
            control: Control input vector.

        Returns:
            Tuple of (A, B) where A is df/dx and B is df/du.
        """
        pass

    @abstractmethod
    def compute_cost(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        ref_path: np.ndarray,
        collision_constraint: List
    ) -> float:
        """Compute the cost of a trajectory.

        Args:
            states: State trajectory.
            controls: Control sequence.
            ref_path: Reference path waypoints.
            collision_constraint: Collision constraints.

        Returns:
            Total cost value.
        """
        pass

    @abstractmethod
    def get_derivates(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        ref_path: np.ndarray,
        collision_constraint: List
    ) -> Tuple:
        """Compute cost derivatives for iLQR backward pass.

        Args:
            states: State trajectory.
            controls: Control sequence.
            ref_path: Reference path waypoints.
            collision_constraint: Collision constraints.

        Returns:
            Tuple of (l_x, l_u, l_xx, l_uu, l_ux) derivatives.
        """
        pass

    @abstractmethod
    def get_derivates_alm(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        ref_path: np.ndarray,
        collision_constraint: List
    ) -> Tuple:
        """Compute ALM cost derivatives for iLQR backward pass.

        Args:
            states: State trajectory.
            controls: Control sequence.
            ref_path: Reference path waypoints.
            collision_constraint: Collision constraints.

        Returns:
            Tuple of (l_x, l_u, l_xx, l_uu, l_ux) derivatives.
        """
        pass

    @abstractmethod
    def compute_cost_alm(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        ref_path: np.ndarray,
        collision_constraint: List
    ) -> float:
        """Compute ALM-augmented cost of a trajectory.

        Args:
            states: State trajectory.
            controls: Control sequence.
            ref_path: Reference path waypoints.
            collision_constraint: Collision constraints.

        Returns:
            Total ALM-augmented cost value.
        """
        pass

    @abstractmethod
    def compute_constraint(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        obstacles: List,
        step: int
    ) -> np.ndarray:
        """Compute constraint values at a specific timestep.

        Args:
            states: State trajectory.
            controls: Control sequence.
            obstacles: Obstacle objects.
            step: Timestep index (1 to horizon).

        Returns:
            Constraint values (positive means violation).
        """
        pass

    @abstractmethod
    def _projection(self, x: np.ndarray) -> np.ndarray:
        """Project vector onto non-positive orthant.

        Args:
            x: Input vector.

        Returns:
            Projected vector (max(x, 0) becomes 0).
        """
        pass

    @abstractmethod
    def get_center(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute front and rear axle center positions.

        Args:
            state: Vehicle state [x, y, v, phi, yaw].

        Returns:
            Tuple of (front_center, rear_center) as 2D position arrays.
        """
        pass

    @abstractmethod
    def get_reference_state(
        self,
        states: np.ndarray,
        ref_path: np.ndarray
    ) -> np.ndarray:
        """Find reference states for each trajectory point.

        Args:
            states: State trajectory.
            ref_path: Reference path waypoints.

        Returns:
            Reference states matching the trajectory shape.
        """
        pass

    @abstractmethod
    def init_traj(
        self,
        init_state: np.ndarray,
        controls: np.ndarray,
        horizon: int = 60
    ) -> np.ndarray:
        """Initialize state trajectory by forward simulation.

        Args:
            init_state: Initial state vector.
            controls: Control sequence.
            horizon: Planning horizon length.

        Returns:
            State trajectory of shape (horizon+1, state_dim).
        """
        pass

    def exp_barrier(self, c: float, q1: float = 5.5, q2: float = 5.75) -> float:
        """Compute exponential barrier function.

        Args:
            c: Constraint value (positive means violation).
            q1: First exponential parameter.
            q2: Second exponential parameter.

        Returns:
            Barrier function value.
        """
        b = q1 * np.exp(q2 * c)
        return b

    def exp_barrier_derivative_and_hessian(
        self,
        c: float,
        c_dot: np.ndarray,
        q1: float = 5.5,
        q2: float = 5.75
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute exponential barrier derivative and Hessian.

        Args:
            c: Constraint value.
            c_dot: Constraint derivative vector.
            q1: First exponential parameter.
            q2: Second exponential parameter.

        Returns:
            Tuple of (b_dot, b_ddot) derivative and Hessian.
        """
        b = self.exp_barrier(c, q1, q2)
        b_dot = q2 * b * c_dot
        c_dot_reshaped = c_dot[:, np.newaxis]
        b_ddot = (q2 ** 2) * b * (c_dot_reshaped @ c_dot_reshaped.T)

        return b_dot, b_ddot

    def get_bound_constr(
        self,
        var: float,
        bound: float,
        bound_type: str = 'upper'
    ) -> float:
        """Compute bound constraint value.

        Args:
            var: Variable value.
            bound: Bound limit.
            bound_type: Either 'upper' or 'lower'.

        Returns:
            Constraint value (positive means violation).
        """
        assert bound_type == 'upper' or bound_type == 'lower'
        if bound_type == 'upper':
            return var - bound
        else:  # lower
            return bound - var

    def alm_term(self, c: float, lamb: float, mu: float) -> float:
        """Compute ALM penalty term.

        Args:
            c: Constraint value.
            lamb: Lagrange multiplier.
            mu: Penalty parameter.

        Returns:
            ALM penalty value.
        """
        return lamb * c + 0.5 * c * mu * c

    def alm_derivative_and_hessian(
        self,
        c: np.ndarray,
        c_dot: np.ndarray,
        lamb: float,
        mu: float
    ) -> Tuple:
        """Compute ALM derivative and Hessian (placeholder).

        Args:
            c: Constraint value.
            c_dot: Constraint derivative.
            lamb: Lagrange multiplier.
            mu: Penalty parameter.

        Returns:
            Tuple of (derivative, Hessian).
        """
        pass

    def project(self, var: float, bound: float, bound_type: str = 'upper'):
        """Project variable onto feasible set.

        Args:
            var: Variable value.
            bound: Bound limit.
            bound_type: Either 'upper' or 'lower'.
        """
        pass

    def init_multipliers(self, constraint_dim: int) -> None:
        """Initialize ALM multipliers and penalty matrices.

        Args:
            constraint_dim: Dimension of the constraint vector.
        """
        self.mu = 1.0
        self.i_mu = np.stack(
            [np.eye(constraint_dim) for _ in range(self.horizon)], axis=0
        )
        self.lambda_alm = np.zeros((self.horizon, constraint_dim))

    def update_lambda(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        obstacles: List
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
        obstacles: List
    ) -> None:
        """Update penalty matrix I_mu based on active constraints.

        Only constraints with positive multiplier or violation are penalized.

        Args:
            states: State trajectory.
            controls: Control sequence.
            obstacles: List of obstacle objects.
        """
        if not hasattr(self, 'i_mu'):
            constraint_dim = 2 * self.control_dim + 2 + 2 * len(obstacles)
            self.init_multipliers(constraint_dim)

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
