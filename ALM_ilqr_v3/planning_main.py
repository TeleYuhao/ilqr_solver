"""Main script for single-shot ALM iLQR trajectory planning with visualization.

This script demonstrates the ALM iLQR solver for a single planning problem with
static obstacles. It computes an optimal trajectory from an initial state,
visualizes the vehicle motion at each timestep, and displays both the actual
obstacle size (rectangle) and safety boundary (ellipse).

Example:
    >>> python planning_main.py
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from alm_ilqr_core import ALMILQRCore
from ALM_Model_v2 import ALMModelV2
from obstacle import Obstacle


# ==================== Vehicle Parameters ====================
VEHICLE_LENGTH = 4.5  # Vehicle length [m]
VEHICLE_WIDTH = 2.0  # Vehicle width [m]
WHEELBASE = 3.6  # Wheelbase [m]
SAFETY_BUFFER = 1.5  # Safety buffer [m]

# ==================== Configuration ====================
CONFIG = {
    'v_max': 10.0,
    'v_min': 0.0,
    'acc_max': 2.0,
    'acc_min': -2.0,
    'omega_max': 3.14,
    'omega_min': -3.14,
    'horizon': 60,
    'dt': 0.1,
    'Q': np.diag([0.0, 1.0, 0.5, 0.0, 0.0]),
    'R': np.diag([1.0, 1.0]),
    'ref_velo': 6.0,
    'state_dim': 5,
    'control_dim': 2,
    'wheelbase': WHEELBASE,
    'max_iter': 20,
    'tol': 1e-5,
    'lamb_decay': 0.6,
    'lamb_amplify': 2.0,
    'max_lamb': 1e4,
    'armijo_c': 0.05,
    'armijo_beta': 0.5,
    'armijo_alpha_min': 1e-1,
    'init_lamb': 20.0,
    'max_alm_iters': 10,
    'max_ilqr_iters': 10,
    'violation_tol': 1e-7,
    'small_violation_threshold': 1e-2,
    'mu_gain': 8,
}


def get_vehicle_corners(
    x: float,
    y: float,
    yaw: float,
    length: float = VEHICLE_LENGTH,
    width: float = VEHICLE_WIDTH
) -> np.ndarray:
    """Calculate the four corners of the vehicle rectangle.

    Args:
        x: Vehicle center x-coordinate.
        y: Vehicle center y-coordinate.
        yaw: Vehicle heading angle in radians.
        length: Vehicle length in meters.
        width: Vehicle width in meters.

    Returns:
        4x2 array of corner coordinates [x, y].
        Order: front-left, front-right, rear-right, rear-left.
    """
    half_length = length / 2.0
    half_width = width / 2.0

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation_matrix = np.array([[cos_yaw, -sin_yaw],
                                [sin_yaw, cos_yaw]])

    corners_local = np.array([
        [half_length, half_width],   # front-left
        [half_length, -half_width],  # front-right
        [-half_length, -half_width],  # rear-right
        [-half_length, half_width]   # rear-left
    ])

    corners_global = (rotation_matrix @ corners_local.T).T + np.array([x, y])
    return corners_global


def run_planning() -> Tuple[np.ndarray, np.ndarray, list, list]:
    """Run the ALM iLQR trajectory planning.

    Plans an optimal trajectory avoiding static obstacles with two scenarios:
    1. Initial planning from origin
    2. Warm start from a partially executed trajectory

    Returns:
        Tuple of (x, u, trajectory_history, cost_history) where:
            x: Optimized state trajectory.
            u: Optimized control sequence.
            trajectory_history: List of state trajectories from each iteration.
            cost_history: List of cost values from each iteration.
    """
    # Initial state [x, y, v, phi, yaw]
    ego_state = [0.0, 0.0, 5.0, 0.0, 0]

    # Reference path (straight line)
    longit_ref = np.linspace(0, 50, 1000)
    lateral_ref = np.linspace(0, 0, 1000)
    ref_waypoints = np.vstack((longit_ref, lateral_ref))
    ref_velo = np.array(6.0)

    # Obstacle attributes [width, length, safety_buffer]
    obstacle_attr_1 = np.array([VEHICLE_WIDTH, VEHICLE_LENGTH, SAFETY_BUFFER])
    obstacle_attr_2 = np.array([VEHICLE_WIDTH, VEHICLE_LENGTH, SAFETY_BUFFER])

    # Obstacle states [x, y, v, yaw]
    obs_1 = [8, -0.2, 3, -0.0]
    obs_2 = [20, 4, -2.0, 0]

    obstacle_list = [
        Obstacle(obs_1, obstacle_attr_1),
        Obstacle(obs_2, obstacle_attr_2)
    ]

    # Create model
    model = ALMModelV2(CONFIG)
    ego_state = [0.0, 0.0, 5.0, 0.0, 0]  # [x, y, v, phi, yaw]

    init_control = np.zeros((60, 2))
    init_x = model.init_traj(ego_state, init_control)

    # First planning pass
    solver = ALMILQRCore(model, CONFIG)
    x, u, trajectory_history, cost_history = solver.solve(
        init_x, init_control, ref_waypoints, obstacle_list
    )

    # Warm start from partially executed trajectory
    # obs_1 = [9.3, -0.2, 1.0, -0.0]
    # obs_2 = [20.6, 4.0, 2.0, 0.0]
    # ego_traj = np.array([1.52072896, 0.01206965, 5.20649835, 0.38663576, 0.06680481])
    # init_x = model.init_traj(ego_traj, u)

    # obs_list = [
    #     Obstacle(obs_1, obstacle_attr_1),
    #     Obstacle(obs_2, obstacle_attr_2)
    # ]
    # x, u, trajectory_history, cost_history = solver.solve(
    #     init_x, init_control, ref_waypoints, obs_list
    # )

    print('Solver completed!')
    print(f'Final cost: {cost_history[-1]:.4f}')
    print(f'Number of iterations: {len(cost_history)}')

    return x, u, trajectory_history, cost_history


def visualize_trajectory(
    x: np.ndarray,
    ref_waypoints: np.ndarray,
    obstacle_list: list
) -> None:
    """Visualize the vehicle trajectory with obstacles.

    Args:
        x: Optimized state trajectory of shape (horizon+1, state_dim).
        ref_waypoints: Reference path waypoints.
        obstacle_list: List of obstacle objects.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot reference path
    ax.plot(ref_waypoints[0], ref_waypoints[1], 'k--',
            alpha=0.5, linewidth=1.5, zorder=1)

    # Plot obstacles
    for obs in obstacle_list:
        # Safety boundary ellipse (dashed line)
        safety_ellipse = plt.matplotlib.patches.Ellipse(
            (obs.pos[0], obs.pos[1]), 2 * obs.a, 2 * obs.b,
            angle=np.degrees(obs.yaw), fill=False,
            edgecolor='red', linewidth=2, linestyle='--', alpha=0.7
        )
        ax.add_patch(safety_ellipse)

        # Actual obstacle size (rectangle)
        obs_corners = get_vehicle_corners(
            obs.pos[0], obs.pos[1], obs.yaw,
            length=obs.obs_length, width=obs.obs_width
        )
        obs_rect = Polygon(
            obs_corners, closed=True,
            facecolor='red', edgecolor='darkred',
            linewidth=2, alpha=0.7, zorder=4
        )
        ax.add_patch(obs_rect)

    # Get final trajectory
    final_traj = x
    n_steps = len(final_traj)

    # Plot vehicle rectangles
    for i in range(n_steps):
        state = final_traj[i]
        x_pos, y_pos, v, phi, yaw = state

        corners = get_vehicle_corners(x_pos, y_pos, yaw)

        # Color gradient (blue -> cyan -> green -> yellow -> red)
        color_val = i / (n_steps - 1) if n_steps > 1 else 0
        color = plt.cm.turbo(color_val)

        if i == 0:
            # Start position
            poly = Polygon(
                corners, closed=True, facecolor='blue', edgecolor='black',
                linewidth=2, alpha=0.9, label='Start', zorder=5
            )
        elif i == n_steps - 1:
            # End position
            poly = Polygon(
                corners, closed=True, facecolor='red', edgecolor='black',
                linewidth=2, alpha=0.9, label='End', zorder=5
            )
        else:
            # Intermediate positions
            poly = Polygon(
                corners, closed=True, facecolor=color, edgecolor='black',
                linewidth=0.5, alpha=0.5, zorder=3
            )

        ax.add_patch(poly)

    # Plot trajectory center line
    ax.plot(final_traj[:, 0], final_traj[:, 1], 'k-',
            linewidth=1.5, alpha=0.8, zorder=2)

    # Direction arrows at start and end
    ax.arrow(final_traj[0, 0], final_traj[0, 1],
             2.0 * np.cos(final_traj[0, 4]),
             2.0 * np.sin(final_traj[0, 4]),
             head_width=0.4, head_length=0.3,
             fc='blue', ec='blue', linewidth=2, zorder=10)
    ax.arrow(final_traj[-1, 0], final_traj[-1, 1],
             2.0 * np.cos(final_traj[-1, 4]),
             2.0 * np.sin(final_traj[-1, 4]),
             head_width=0.4, head_length=0.3,
             fc='red', ec='red', linewidth=2, zorder=10)

    # Legend
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='--', label='Reference path'),
        Line2D([0], [0], color='black', linewidth=1.5, label='Trajectory'),
        Polygon([[0, 0]], closed=True, facecolor='red', edgecolor='darkred',
                alpha=0.7, label='Obstacle (actual)'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2,
               label='Safety boundary'),
        Polygon([[0, 0]], closed=True, facecolor='blue', edgecolor='black',
                alpha=0.9, label='Start'),
        Polygon([[0, 0]], closed=True, facecolor='red', edgecolor='black',
                alpha=0.9, label='End'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Plot properties
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Vehicle Trajectory Evolution (n={n_steps} steps)',
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Auto-scale
    ax.relim()
    ax.autoscale_view()

    # Colorbar for time progression
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.turbo, norm=plt.Normalize(vmin=0, vmax=n_steps - 1)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time Step', fontsize=11)

    plt.tight_layout()
    plt.savefig('/home/yuhao/Code/te/ALM_ilqr_v3/vehicle_trajectory.png',
                dpi=300, bbox_inches='tight')
    print('\nVisualization saved to: '
          '/home/yuhao/Code/te/ALM_ilqr_v3/vehicle_trajectory.png')
    plt.show()


def main() -> None:
    """Main entry point for trajectory planning and visualization."""
    # Run planning
    x, u, trajectory_history, cost_history = run_planning()

    # Reference path and obstacles for visualization
    longit_ref = np.linspace(0, 50, 1000)
    lateral_ref = np.linspace(0, 0, 1000)
    ref_waypoints = np.vstack((longit_ref, lateral_ref))

    obstacle_attr_1 = np.array([VEHICLE_WIDTH, VEHICLE_LENGTH, SAFETY_BUFFER])
    obstacle_attr_2 = np.array([VEHICLE_WIDTH, VEHICLE_LENGTH, SAFETY_BUFFER])

    obs_1 = [9.3, -0.2, 1.0, -0.0]
    obs_2 = [20.6, 4.0, 2.0, 0.0]

    obstacle_list = [
        Obstacle(obs_1, obstacle_attr_1),
        Obstacle(obs_2, obstacle_attr_2)
    ]

    # Visualize
    visualize_trajectory(x, ref_waypoints, obstacle_list)


if __name__ == '__main__':
    main()
