"""Obstacle class with ellipsoidal safety margins for collision avoidance.

This module defines the obstacle class used for trajectory optimization.
Obstacles are represented as ellipsoidal safety regions with constant-velocity
prediction for future motion planning.

The safety margin is computed as:
    margin = 1 - (x_std^2/a^2 + y_std^2/b^2)
where (a, b) are the ellipsoid semi-axes and (x_std, y_std) are the
obstacle position in the ellipsoid's coordinate frame.

Example:
    >>> state = [10, -0.2, 1, 0]  # [x, y, v, yaw]
    >>> attr = [2.0, 4.5, 1.5]  # [width, length, safety_buffer]
    >>> obs = Obstacle(state, attr)
    >>> safety = obs.ellipsoid_safety_margin(ego_pos, obs.pos)
"""

from typing import Tuple

import numpy as np


# Vehicle parameter constants
EGO_WIDTH = 2.0  # meters
EGO_PNT_RADIUS = EGO_WIDTH / 2.0  # meters


class Obstacle:
    """Obstacle with ellipsoidal safety margin for collision avoidance.

    Attributes:
        state: Obstacle state [x, y, v, yaw].
        attr: Obstacle attributes [width, length, safety_buffer].
        pos: Obstacle 2D position [x, y].
        yaw: Obstacle heading angle in radians.
        ego_width: Ego vehicle width (for safety margin calculation).
        ego_pnt_radius: Ego vehicle point approximation radius.
        obs_width: Obstacle width in meters.
        obs_length: Obstacle length in meters.
        d_safe: Safety buffer distance in meters.
        a: Ellipsoid semi-major axis (length direction).
        b: Ellipsoid semi-minor axis (width direction).
        prediction_traj: Predicted trajectory over horizon.
    """

    def __init__(self, state: np.ndarray, attr: np.ndarray) -> None:
        """Initialize obstacle.

        Args:
            state: Obstacle state [x, y, v, yaw] (list or numpy array).
            attr: Obstacle attributes [width, length, safety_buffer] (list or numpy array).
        """
        # Ensure state and attr are numpy arrays (handles list inputs)
        self.state = np.array(state, dtype=float)
        self.attr = np.array(attr, dtype=float)
        self.pos = np.array([self.state[0], self.state[1]])
        self.yaw = self.state[3]
        self.ego_width = EGO_WIDTH
        self.ego_pnt_radius = self.ego_width / 2.0

        self.obs_width = attr[0]
        self.obs_length = attr[1]
        self.d_safe = attr[2]
        self._get_ellipsoid_obstacle_scales()
        self.const_velo_prediction(self.state, 60)

    def _get_ellipsoid_obstacle_scales(self) -> None:
        """Compute ellipsoid semi-axes from dimensions and safety buffer."""
        self.a = (0.5 * self.obs_length + self.d_safe +
                  self.ego_pnt_radius)
        self.b = (0.5 * self.obs_width + self.d_safe +
                  self.ego_pnt_radius)

    def ellipsoid_safety_margin(
        self,
        pnt: np.ndarray,
        elp_center: np.ndarray
    ) -> float:
        """Compute ellipsoid safety margin for a point.

        Args:
            pnt: Point to check [x, y].
            elp_center: Ellipsoid center [x, y].

        Returns:
            Safety margin (positive means safe, negative means collision).
        """
        theta = self.yaw

        diff = pnt - elp_center
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        pnt_std = diff @ rotation_matrix  # Rotate by (-theta)

        result = 1 - ((pnt_std[0] ** 2) / (self.a ** 2) +
                      (pnt_std[1] ** 2) / (self.b ** 2))

        return result

    def ellipsoid_safety_margin_derivatives(
        self,
        pnt: np.ndarray,
        elp_center: np.ndarray
    ) -> np.ndarray:
        """Compute derivative of safety margin w.r.t. point position.

        Args:
            pnt: Point to check [x, y].
            elp_center: Ellipsoid center [x, y].

        Returns:
            Gradient vector d(margin)/d(pnt) of shape (2,).
        """
        theta = self.yaw
        diff = pnt - elp_center
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        pnt_std = diff @ rotation_matrix

        # Constraint over standard point vector: [c -> x_std, c -> y_std]
        res_over_pnt_std = np.array([
            -2 * pnt_std[0] / (self.a ** 2),
            -2 * pnt_std[1] / (self.b ** 2)
        ])

        # Standard point vector over difference vector
        pnt_std_over_diff = rotation_matrix.transpose()

        # Difference vector over original point vector (identity)
        diff_over_pnt = np.eye(2)

        # Chain rule: [c -> x, c -> y]
        res_over_pnt = res_over_pnt_std @ pnt_std_over_diff @ diff_over_pnt

        return res_over_pnt

    @staticmethod
    def kinematic_propagate(x0: np.ndarray) -> np.ndarray:
        """Propagate obstacle state forward by one timestep.

        Uses constant velocity model.

        Args:
            x0: Current state [x, y, v, yaw].

        Returns:
            Next state after dt=0.1 seconds.
        """
        x, y, v, yaw = x0
        dt = 0.1
        return np.array([
            x + v * np.cos(yaw) * dt,
            y + v * np.sin(yaw) * dt,
            v,
            yaw
        ])

    def const_velo_prediction(
        self,
        x0: np.ndarray,
        steps: int
    ) -> np.ndarray:
        """Generate constant-velocity prediction trajectory.

        Args:
            x0: Initial state [x, y, v, yaw].
            steps: Number of prediction steps.

        Returns:
            Predicted trajectory of shape (steps+1, 4).
        """
        predicted_states = [x0]
        cur_x = x0
        for _ in range(steps):
            next_x = self.kinematic_propagate(cur_x)
            cur_x = next_x
            predicted_states.append(next_x)

        predicted_states = np.vstack(predicted_states)
        self.prediction_traj = predicted_states
        return predicted_states
