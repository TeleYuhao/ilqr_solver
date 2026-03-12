"""ALM iLQR solver for trajectory optimization.

This module implements a two-loop Augmented Lagrangian Method (ALM) combined with
the Iterative Linear Quadratic Regulator (iLQR) algorithm for constrained trajectory
optimization. The solver uses a bicycle kinematic model and handles obstacle
avoidance, velocity bounds, and control constraints.

The ALM framework consists of:
- Outer loop: Updates Lagrange multipliers (lambda) and penalty weights (mu)
             based on constraint violation
- Inner loop: Runs iLQR optimization to convergence with fixed multipliers

Example:
    >>> model = ALM_Model(config)
    >>> solver = ALMILQRCore(model, config)
    >>> x_opt, u_opt, traj_hist, cost_hist = solver.solve(
    ...     init_x, init_u, ref_path, obstacles
    ... )
"""

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
from model_base import ModelBase

if TYPE_CHECKING:
    from alm_model import ALMModel


class ALMILQRCore:
    """Two-loop ALM iLQR solver for trajectory optimization.

    The solver implements an augmented Lagrangian method where the outer loop
    enforces constraint satisfaction by updating multipliers, and the inner
    loop performs iLQR optimization with regularized backward/forward passes.

    Attributes:
        model: The kinematic model for forward simulation and derivatives.
        max_iter: Maximum iLQR iterations per inner loop.
        tol: Cost convergence tolerance.
        lamb_decay: Regularization decay factor on successful iteration.
        lamb_amplify: Regularization amplification on failed iteration.
        max_lamb: Maximum regularization parameter.
        armijo_c: Armijo sufficient decrease constant.
        armijo_beta: Alpha shrinkage factor for line search.
        armijo_alpha_min: Minimum step size for line search.
        init_lamb: Initial regularization parameter.
        max_alm_iters: Maximum outer loop iterations.
        max_ilqr_iters: Maximum inner loop iterations.
        violation_tol: Constraint violation tolerance for convergence.
        small_violation_threshold: Threshold for "small" violation.
        mu_gain: Multiplicative factor for increasing mu.
        horizon: Planning horizon length.
        state_dim: State vector dimension.
        control_dim: Control vector dimension.
    """

    def __init__(self, model: ModelBase, config: Dict) -> None:
        """Initialize the ALM iLQR solver.

        Args:
            model: Kinematic model implementing forward_calculation and
                   get_jacobian methods.
            config: Dictionary containing solver configuration parameters.
        """
        self.model          = model
        self.max_iter       = config['max_iter']
        self.tol            = config['tol']
        self.lamb_decay     = config['lamb_decay']
        self.lamb_amplify   = config['lamb_amplify']
        self.max_lamb       = config['max_lamb']
        self.armijo_c       = config['armijo_c']
        self.armijo_beta    = config['armijo_beta']
        self.armijo_alpha_min   = config['armijo_alpha_min']
        self.init_lamb          = config['init_lamb']
        self.max_alm_iters      = config['max_alm_iters']
        self.max_ilqr_iters     = config['max_ilqr_iters']
        self.violation_tol      = config['violation_tol']
        self.small_violation_threshold = config['small_violation_threshold']
        self.mu_gain = config['mu_gain']

        self.horizon    = model.horizon
        self.state_dim  = model.state_dim
        self.control_dim = model.control_dim

    def forwardpass(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        d: np.ndarray,
        K: np.ndarray,
        alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute forward pass with feedback/feedforward correction.

        Args:
            states: Nominal state trajectory of shape (horizon+1, state_dim).
            controls: Nominal control sequence of shape (horizon, control_dim).
            d: Feedforward gains of shape (horizon, control_dim).
            K: Feedback gains of shape (horizon, control_dim, state_dim).
            alpha: Line search step size.

        Returns:
            Tuple of (new_u, new_x) containing:
                - new_u: Corrected control sequence.
                - new_x: Resulting state trajectory.
        """
        new_u = np.zeros((self.horizon, self.control_dim))
        new_x = np.zeros((self.horizon + 1, self.state_dim))
        new_x[0] = states[0]

        for i in range(self.horizon):
            new_u_i = controls[i] + alpha * d[i] + K[i] @ (
                new_x[i] - states[i]
            )
            new_u[i] = new_u_i
            new_x[i + 1] = self.model.forward_calculation(new_x[i], new_u[i])
        return new_u, new_x

    def backwardpass(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        ref_path: np.ndarray,
        obstacles: List,
        lamb: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Execute backward pass to compute feedback gains.

        Args:
            states: Current state trajectory.
            controls: Current control sequence.
            ref_path: Reference waypoints.
            obstacles: List of obstacle objects.
            lamb: Regularization parameter.

        Returns:
            Tuple of (d, K, delta_V) containing:
                - d: Feedforward gains.
                - K: Feedback gains.
                - delta_V: Expected cost reduction.
        """
        l_x, l_u, l_xx, l_uu, l_ux = self.model.get_derivates_alm(
            states, controls, ref_path, obstacles
        )
        delta_v = 0
        v_x = l_x[-1]
        v_xx = l_xx[-1]
        d = np.zeros((self.horizon, self.control_dim))
        k = np.zeros((self.horizon, self.control_dim, self.state_dim))

        regu_i = lamb * np.eye(v_xx.shape[0])

        for i in reversed(range(self.horizon)):
            dfdx, dfdu = self.model.get_jacobian(states[i], controls[i])

            q_x = l_x[i] + dfdx.T @ v_x
            q_u = l_u[i] + dfdu.T @ v_x
            q_xx = l_xx[i] + dfdx.T @ v_xx @ dfdx
            q_uu = l_uu[i] + dfdu.T @ v_xx @ dfdu
            q_ux = l_ux[i] + dfdu.T @ v_xx @ dfdx

            df_du_regu = dfdu.T @ regu_i
            q_ux_regu = q_ux + df_du_regu @ dfdx
            q_uu_regu = q_uu + df_du_regu @ dfdu
            q_uu_inv = np.linalg.inv(q_uu_regu)

            d[i] = -q_uu_inv @ q_u
            k[i] = -q_uu_inv @ q_ux_regu

            v_x = q_x + k[i].T @ q_uu @ d[i] + k[i].T @ q_u + q_ux.T @ d[i]
            v_xx = (q_xx + k[i].T @ q_uu @ k[i] +
                    k[i].T @ q_ux + q_ux.T @ k[i])

            delta_v += 0.5 * d[i].T @ q_uu @ d[i] + d[i].T @ q_u

        return d, k, delta_v

    def iterate(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        j: float,
        ref_path: np.ndarray,
        obstacles: List,
        lamb: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """Perform one iLQR iteration with line search.

        Args:
            states: Current state trajectory.
            controls: Current control sequence.
            j: Current cost value.
            ref_path: Reference waypoints.
            obstacles: List of obstacle objects.
            lamb: Regularization parameter.

        Returns:
            Tuple of (new_u, new_x, new_j, success) containing:
                - new_u: Updated control sequence.
                - new_x: Updated state trajectory.
                - new_j: Updated cost value.
                - success: Whether iteration was accepted.
        """
        j = self.model.compute_cost_alm(states, controls, ref_path, obstacles)
        d, k, expc_red = self.backwardpass(
            states, controls, ref_path, obstacles, lamb
        )

        alpha = 1.0
        accept = False
        new_u, new_x, new_j = None, None, float('inf')

        while alpha > self.armijo_alpha_min:
            new_u, new_x = self.forwardpass(states, controls, d, k, alpha)
            new_j = self.model.compute_cost_alm(
                new_x, new_u, ref_path, obstacles
            )

            expected_reduction = alpha * expc_red

            if new_j <= j - self.armijo_c * expected_reduction:
                accept = True
                print(f'    Armijo accepted: alpha={alpha:.4f}, '
                      f'new_J={new_j:.4f}, '
                      f'expected_reduction={expected_reduction:.4f}')
                break
            elif alpha < self.armijo_alpha_min:
                print(f'    Armijo rejected, alpha < alpha_min: {alpha:.4f}')
                break
            else:
                alpha *= self.armijo_beta
                print(f'    Armijo rejected, shrinking alpha to {alpha:.4f}')

        if not accept:
            print('    Line search failed: alpha < alpha_min')
            return states, controls, j, False

        return new_u, new_x, new_j, True

    def solve(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        ref_path: np.ndarray,
        obstacles: List
    ) -> Tuple[np.ndarray, np.ndarray, List, List]:
        """Solve the ALM iLQR optimization problem.

        Two-loop algorithm:
        - Outer loop: Update lambda/mu based on constraint violation
        - Inner loop: Run iLQR to convergence

        Args:
            states: Initial state trajectory of shape (horizon+1, state_dim).
            controls: Initial control sequence of shape (horizon, control_dim).
            ref_path: Reference waypoints for tracking.
            obstacles: List of obstacle objects for avoidance.

        Returns:
            Tuple of (x, u, trajectory_history, cost_history) containing:
                - x: Optimized state trajectory.
                - u: Optimized control sequence.
                - trajectory_history: List of state trajectories per iteration.
                - cost_history: List of cost values per iteration.

        Raises:
            ValueError: If controls shape doesn't match horizon.
        """
        if controls.shape[0] != self.horizon:
            raise ValueError(
                f"controls shape {controls.shape} doesn't match horizon "
                f"{self.horizon}"
            )

        constraint_dim = 2 * self.model.control_dim + 2 + 2 * len(obstacles)
        self.model.init_multipliers(constraint_dim)
        u, x = controls, states
        trajectory_history = [x.copy()]
        cost_history = [
            self.model.compute_cost_alm(x, u, ref_path, obstacles)
        ]

        print("=" * 60)
        print("Two-Loop ALM iLQR Solver")
        print("=" * 60)

        for alm_iter in range(self.max_alm_iters):
            print(f"\n=== ALM Iteration {alm_iter} ===")

            lamb = self.init_lamb
            ilqr_converged = False

            for ilqr_iter in range(self.max_ilqr_iters):
                new_u, new_x, new_j, iter_effective_flag = self.iterate(
                    x, u, cost_history[-1], ref_path, obstacles, lamb
                )

                if iter_effective_flag:
                    x = new_x
                    u = new_u

                    trajectory_history.append(x.copy())
                    cost_history.append(new_j)

                    if abs(cost_history[-2] - new_j) < self.tol:
                        print(f"  iLQR converged at iteration {ilqr_iter}")
                        ilqr_converged = True
                        break

                    lamb *= self.lamb_decay
                else:
                    lamb *= self.lamb_amplify
                    if lamb > self.max_lamb:
                        print("  Regularization max reached, stopping iLQR")
                        break

            violation = self.model.compute_violation(x, u, obstacles)
            print(f"  Constraint violation: {violation:.6f}, "
                  f"mu: {self.model.mu:.4f}")

            if violation < self.violation_tol:
                print(f"  ✓ Converged! Violation < {self.violation_tol}")
                break
            elif violation < self.small_violation_threshold:
                print(f"  → Small violation - updating lambda only")
                self.model.update_lambda(x, u, obstacles)
            else:
                print(f"  → Large violation - increasing mu by 2x")
                self.model.update_mu_scalar(self.mu_gain)
                self.model.update_lambda(x, u, obstacles)

        print("\n" + "=" * 60)
        print(f"Final violation: {self.model.compute_violation(x, u, obstacles):.6f}")
        print(f"Final mu: {self.model.mu:.4f}")
        print("=" * 60)

        return x, u, trajectory_history, cost_history
