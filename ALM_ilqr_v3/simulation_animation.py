#!/usr/bin/env python3
"""Dynamic simulation and animation generation for ALM iLQR trajectory planning.

This script simulates an autonomous vehicle navigating around moving obstacles
using Model Predictive Control (MPC) with ALM iLQR. At each timestep, the vehicle
re-plans its trajectory considering the predicted motion of obstacles. The simulation
generates an animated GIF showing the vehicle, obstacles, planned trajectories,
and executed path.

Example:
    >>> sim_history, ref_path, failures = run_simulation()
    >>> ani = create_animation(sim_history, ref_path)
    >>> plt.show()
"""

import copy
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np

from alm_ilqr_core import ALMILQRCore
# from alm_model import ALMModel
from ALM_Model_v2 import ALMModelV2 as ALMModel
from obstacle import Obstacle


# ==================== Vehicle Parameters ====================
VEHICLE_LENGTH = 4.5  # Vehicle length [m]
VEHICLE_WIDTH = 2.0  # Vehicle width [m]
WHEELBASE = 3.6  # Wheelbase [m]
SAFETY_BUFFER = 1.0  # Safety buffer [m]

# ==================== Configuration ====================
CONFIG: Dict = {
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
    'ref_velo': 8.0,
    'state_dim': 5,
    'control_dim': 2,
    'wheelbase': WHEELBASE,
    'max_iter': 15,
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
    'violation_tol': 1e-4,
    'small_violation_threshold': 1e-2,
    'mu_gain': 8,
}

# Simulation parameters
DT = 0.1  # Time step [s]
SIM_DURATION = 6.0  # Simulation duration [s]
N_STEPS = int(SIM_DURATION / DT)  # Number of simulation steps


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


def create_obstacle_copy(obs: Obstacle) -> Obstacle:
    """Create a deep copy of an obstacle for simulation.

    Args:
        obs: Original obstacle object.

    Returns:
        New obstacle object with copied state and attributes.
        The prediction_traj is recomputed based on the new state.
    """
    new_obs = Obstacle(obs.state.copy(), obs.attr.copy())
    return new_obs


def run_simulation() -> Tuple[Dict, np.ndarray, List[Dict]]:
    """Run the complete dynamic simulation.

    The simulation performs MPC-style replanning at each timestep:
    1. Record current vehicle and obstacle states
    2. Replan trajectory using current state and obstacle predictions
    3. Execute first control step
    4. Propagate obstacles forward with constant velocity model

    Returns:
        Tuple of (sim_history, ref_waypoints, planning_failures) where:
            - sim_history: Dictionary containing vehicle_states, obstacle_states,
              time, and planned_trajectories.
            - ref_waypoints: Reference path waypoints.
            - planning_failures: List of failure records with detailed state info.
    """
    print('=' * 60)
    print('ALM iLQR Dynamic Simulation')
    print('=' * 60)

    # ==================== Initialization ====================
    # Reference path (straight line along x-axis)
    longit_ref = np.linspace(0, 40, 1000)
    lateral_ref = np.linspace(0, 0, 1000)
    ref_waypoints = np.vstack((longit_ref, lateral_ref))

    # Vehicle initial state [x, y, v, phi, yaw]
    ego_state = [0.0, 0.0, 5.0, 0.0, 0]

    # Obstacle attributes [width, length, safety_buffer]
    obstacle_attr_1 = np.array([VEHICLE_WIDTH, VEHICLE_LENGTH, SAFETY_BUFFER])
    obstacle_attr_2 = np.array([VEHICLE_WIDTH, VEHICLE_LENGTH, SAFETY_BUFFER])

    # Obstacle states [x, y, v, yaw]
    obs_1 = [8, -0.2, 3, -0.0]
    obs_2 = [20, 4, 2.0, 0]
    # obs_1 = [8, -0.2, 3, -0.0]
    # obs_2 = [20, 4, -2.0, 0]

    obstacle_list_init = [
        Obstacle(obs_1, obstacle_attr_1),
        Obstacle(obs_2, obstacle_attr_2)
    ]

    # Create model
    model = ALMModel(CONFIG)

    # ==================== Simulation Loop ====================
    sim_history = {
        'vehicle_states': [],
        'obstacle_states': [],
        'time': [],
        'planned_trajectories': [],
    }

    planning_failures: List[Dict] = []

    current_state = np.array(ego_state)
    current_obstacles = [
        create_obstacle_copy(obs) for obs in obstacle_list_init
    ]

    print(f'\nSimulation parameters:')
    print(f'  Duration: {SIM_DURATION}s')
    print(f'  Steps: {N_STEPS}')
    print(f'  Time step: {DT}s')
    print(f'\n{"Step":<6} {"Time[s]":<10} {"PosX[m]":<12} {"PosY[m]":<12} '
          f'{"Vel[m/s]":<12}')

    init_control = np.zeros((CONFIG['horizon'], 2))

    for step in range(N_STEPS):
        # Record current state
        sim_history['vehicle_states'].append(current_state.copy())
        sim_history['obstacle_states'].append(
            [obs.state.copy() for obs in current_obstacles]
        )
        sim_history['time'].append(step * DT)

        # Print progress
        if step % 10 == 0 or step == N_STEPS - 1:
            obs_info = ''
            for i, obs in enumerate(current_obstacles):
                obs_info += f' | Obs{i+1}:({obs.state[0]:.1f},{obs.state[1]:.1f})'
            print(f'{step:<6} {step * DT:<10.2f} {current_state[0]:<12.2f} '
                  f'{current_state[1]:<12.2f} {current_state[2]:<12.2f}'
                  f'{obs_info}')

        # Validate state dimension
        state_dim_valid = len(current_state) == 5
        if not state_dim_valid:
            print('Warning: State dimension incorrect, using default values')
            current_state = np.array([0.0, 0.0, 5.0, 0.0, 0])

        # Initialize trajectory for planning
        try:
            init_x = model.init_traj(current_state, init_control)
        except Exception as e:
            print(f'Warning: Trajectory initialization failed, '
                  f'using zero control: {e}')
            init_x = np.zeros((CONFIG['horizon'] + 1, 5))
            init_x[0] = current_state
            for i in range(1, CONFIG['horizon'] + 1):
                init_x[i] = model.forward_calculation(init_x[i - 1], [0, 0])

        # Replan (using current state and obstacle predictions)
        try:
            solver = ALMILQRCore(model, CONFIG)
            x_opt, u_opt, _, cost_history = solver.solve(
                init_x, init_control, ref_waypoints, current_obstacles
            )
            sim_history['planned_trajectories'].append(x_opt.copy())
            next_control = u_opt[0]
            init_control = u_opt
        except Exception as e:
            print(f'⚠️  [Step {step}] Planning failed: '
                  f'{type(e).__name__}: {e}')
            planning_failures.append({
                'step': step,
                'time': step * DT,
                'vehicle_state': current_state.copy(),
                'obstacle_states': [obs.state.copy() for obs in current_obstacles],
                'error_type': type(e).__name__,
                'error_msg': str(e)
            })
            sim_history['planned_trajectories'].append(None)
            next_control = np.array([0.0, 0.0])

        # Update vehicle state
        try:
            current_state = x_opt[1]
        except Exception as e:
            print(f'Warning: State update failed: {e}')
            # Use simple kinematic model as fallback
            x, y, v, phi, yaw = current_state
            a, omega = next_control
            current_state = np.array([
                x + v * np.cos(yaw) * DT,
                y + v * np.sin(yaw) * DT,
                v + a * DT,
                phi + omega * DT,
                yaw + v * np.tan(phi) / WHEELBASE * DT
            ])

        # Update obstacle positions (constant velocity prediction)
        for obs in current_obstacles:
            obs.state = Obstacle.kinematic_propagate(obs.state)
            # Synchronize pos and yaw for consistency
            obs.pos = np.array([obs.state[0], obs.state[1]])
            obs.yaw = obs.state[3]
            # Recompute prediction trajectory
            obs.const_velo_prediction(obs.state, 60)

    print('\n' + '=' * 60)
    print('Simulation Complete!')
    print('=' * 60)

    # Print planning failure summary
    print('\n' + '=' * 60)
    print('Planning Failure Summary')
    print('=' * 60)
    print(f'Total steps: {N_STEPS}')
    print(f'Failures: {len(planning_failures)}')
    print(f'Success rate: {(N_STEPS - len(planning_failures)) / N_STEPS * 100:.1f}%')

    if planning_failures:
        print(f'\nFailure details:')
        print(f'{"Step":<8} {"Time[s]":<10} {"Vehicle Pos":<25} '
              f'{"Error Type":<20}')
        print('-' * 70)
        for fail in planning_failures:
            pos_str = f'({fail["vehicle_state"][0]:.2f}, {fail["vehicle_state"][1]:.2f})'
            print(f'{fail["step"]:<8} {fail["time"]:<10.2f} {pos_str:<25} '
                  f'{fail["error_type"]:<20}')

        # First failure details
        print(f'\nFirst failure details:')
        first_fail = planning_failures[0]
        print(f'  Step: {first_fail["step"]}')
        print(f'  Time: {first_fail["time"]:.2f}s')
        print(f'  Vehicle state: [x={first_fail["vehicle_state"][0]:.2f}, '
              f'y={first_fail["vehicle_state"][1]:.2f}, '
              f'v={first_fail["vehicle_state"][2]:.2f}, '
              f'phi={first_fail["vehicle_state"][3]:.2f}, '
              f'yaw={first_fail["vehicle_state"][4]:.2f}]')
        print(f'  Obstacle states:')
        for i, obs_state in enumerate(first_fail['obstacle_states']):
            print(f'    Obstacle {i+1}: [x={obs_state[0]:.2f}, '
                  f'y={obs_state[1]:.2f}, v={obs_state[2]:.2f}, '
                  f'yaw={obs_state[3]:.2f}]')
        print(f'  Error type: {first_fail["error_type"]}')
        print(f'  Error message: {first_fail["error_msg"]}')
    else:
        print('\n✓ All steps planned successfully!')

    print('=' * 60)

    return sim_history, ref_waypoints, planning_failures


def create_animation(
    sim_history: Dict,
    ref_waypoints: np.ndarray
) -> animation.FuncAnimation:
    """Create animation and save as GIF.

    Args:
        sim_history: Simulation history dictionary containing vehicle_states,
            obstacle_states, time, and planned_trajectories.
        ref_waypoints: Reference path waypoints.

    Returns:
        Matplotlib FuncAnimation object.
    """
    print('\nGenerating animation...')

    n_steps = len(sim_history['vehicle_states'])

    # Compute plot bounds
    all_x = [state[0] for state in sim_history['vehicle_states']]
    all_y = [state[1] for state in sim_history['vehicle_states']]

    for obs_states in sim_history['obstacle_states']:
        for obs_state in obs_states:
            all_x.append(obs_state[0])
            all_y.append(obs_state[1])

    x_margin = 5
    y_margin = 5
    xmin, xmax = min(all_x) - x_margin, max(all_x) + x_margin
    ymin, ymax = min(all_y) - y_margin, max(all_y) + y_margin

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    def init():
        ax.clear()
        return []

    def update(frame: int):
        ax.clear()

        # Set plot properties
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)

        # Get current frame data
        vehicle_state = sim_history['vehicle_states'][frame]
        obstacle_states = sim_history['obstacle_states'][frame]
        current_time = sim_history['time'][frame]

        # Draw reference path
        ax.plot(ref_waypoints[0], ref_waypoints[1], 'k--',
                label='Reference path', alpha=0.4, linewidth=1.5, zorder=1)

        # Draw executed trajectory
        if frame > 0:
            hist_x = [sim_history['vehicle_states'][i][0] for i in range(frame + 1)]
            hist_y = [sim_history['vehicle_states'][i][1] for i in range(frame + 1)]
            ax.plot(hist_x, hist_y, 'b-', linewidth=2, alpha=0.7, zorder=2,
                    label='Executed Trajectory')

        # Draw planned trajectory
        if sim_history['planned_trajectories'][frame] is not None:
            planned_traj = sim_history['planned_trajectories'][frame]
            plan_x = planned_traj[:, 0]
            plan_y = planned_traj[:, 1]
            ax.plot(plan_x, plan_y, 'g--', linewidth=1.5, alpha=0.8, zorder=3,
                    label='Planned Trajectory')

        # Draw obstacles
        for i, obs_state in enumerate(obstacle_states):
            obs_corners = get_vehicle_corners(
                obs_state[0], obs_state[1], obs_state[3],
                length=VEHICLE_LENGTH, width=VEHICLE_WIDTH
            )
            obs_poly = Polygon(
                obs_corners, closed=True,
                facecolor='red', edgecolor='darkred',
                linewidth=2, alpha=0.7, zorder=4
            )
            ax.add_patch(obs_poly)

            ax.text(obs_state[0], obs_state[1] + 2, f'Obs {i+1}',
                   ha='center', va='bottom', fontsize=9,
                   color='darkred', fontweight='bold')

        # Draw vehicle
        vehicle_corners = get_vehicle_corners(
            vehicle_state[0], vehicle_state[1], vehicle_state[4],
            length=VEHICLE_LENGTH, width=VEHICLE_WIDTH
        )
        vehicle_poly = Polygon(
            vehicle_corners, closed=True,
            facecolor='royalblue', edgecolor='navy',
            linewidth=2, alpha=0.9, zorder=5
        )
        ax.add_patch(vehicle_poly)

        # Draw direction arrow
        arrow_len = 2.0
        ax.arrow(vehicle_state[0], vehicle_state[1],
                arrow_len * np.cos(vehicle_state[4]),
                arrow_len * np.sin(vehicle_state[4]),
                head_width=0.5, head_length=0.4,
                fc='navy', ec='navy', linewidth=2, zorder=6)

        # Title and info
        ax.set_title(f'ALM iLQR Dynamic Simulation\\n'
                    f'Time: {current_time:.2f}s / {SIM_DURATION:.2f}s',
                    fontsize=14, fontweight='bold')

        # State info box
        info_text = f'Vehicle State:\\n'
        info_text += f'  X: {vehicle_state[0]:.2f} m\\n'
        info_text += f'  Y: {vehicle_state[1]:.2f} m\\n'
        info_text += f'  V: {vehicle_state[2]:.2f} m/s\\n'
        info_text += f'  Yaw: {np.degrees(vehicle_state[4]):.1f}°'

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Legend
        legend_elements = [
            Polygon([[0, 0]], closed=True, facecolor='royalblue', edgecolor='navy',
                    label='Ego Vehicle', alpha=0.9),
            Polygon([[0, 0]], closed=True, facecolor='red', edgecolor='darkred',
                    label='Obstacle', alpha=0.7),
            Line2D([0], [0], color='black', linestyle='--', label='Reference Path'),
            Line2D([0], [0], color='blue', linewidth=2, label='Executed Trajectory'),
            Line2D([0], [0], color='green', linestyle='--', linewidth=1.5,
                   label='Planned Trajectory'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        return []

    # Create animation
    print(f'Creating {n_steps} frame animation...')
    ani = animation.FuncAnimation(
        fig, update, frames=n_steps,
        init_func=init, blit=False,
        interval=100, repeat=True
    )

    # Save GIF
    output_path = '/home/yuhao/Code/te/ALM_ilqr_v3/simulation.gif'
    print(f'Saving animation to: {output_path}')
    ani.save(output_path, writer='pillow', fps=10, dpi=100)
    print('Animation saved successfully!')

    return ani


def main() -> None:
    """Main entry point for simulation and animation."""
    # Run simulation
    sim_history, ref_waypoints, planning_failures = run_simulation()

    # Create and save animation
    ani = create_animation(sim_history, ref_waypoints)

    print('\n' + '=' * 60)
    print('Simulation Results:')
    print(f'  Total timesteps: {len(sim_history["vehicle_states"])}')
    print(f'  Simulation duration: {sim_history["time"][-1]:.2f}s')
    print(f'  Final position: ({sim_history["vehicle_states"][-1][0]:.2f}, '
          f'{sim_history["vehicle_states"][-1][1]:.2f})')
    print(f'  Final velocity: {sim_history["vehicle_states"][-1][2]:.2f} m/s')
    print(f'  Planning failures: {len(planning_failures)}')
    print(f'  Animation file: /home/yuhao/Code/te/ALM_ilqr_v3/simulation.gif')
    print('=' * 60)

    # Display animation
    plt.show()


if __name__ == '__main__':
    main()
