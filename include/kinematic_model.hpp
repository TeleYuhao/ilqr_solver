#pragma once

#include "model_base.hpp"
#include <Eigen/Core>
#include <cmath>

namespace ilqr {

/**
 * @brief Kinematic bicycle model for vehicle dynamics
 *
 * Implements a kinematic bicycle model with state [x, y, v, yaw] and control [acceleration, steering].
 * Corresponds to Python's KinematicModel class in kinematic_model.py.
 *
 * State: [x, y, v, yaw] where:
 *   - x, y: position (m)
 *   - v: velocity (m/s)
 *   - yaw: heading angle (rad)
 *
 * Control: [a, delta] where:
 *   - a: acceleration (m/s²)
 *   - delta: steering angle (rad)
 */
class KinematicModel : public Model<double, 4, 2> {
public:
    // Type definitions
    ILQR_PROBLEM_VARIABLES(double, 4, 2)

    // Vehicle parameters [m]
    static constexpr double WHEEL_BASE = 3.6;
    static constexpr double WIDTH = 2.0;
    static constexpr double LENGTH = 4.5;

    /**
     * @brief Constructor
     */
    KinematicModel() : Model<double, 4, 2>(4, 2) {}

    /**
     * @brief Destructor
     */
    ~KinematicModel() override = default;

    /**
     * @brief Compute state transition Jacobian with respect to state (A matrix)
     *
     * Computes df/dx for the kinematic bicycle model.
     *
     * @param state Current state [x, y, v, yaw]
     * @param ctrl Current control [acceleration, steering]
     * @param step Time step (default 0.1)
     * @return A matrix (4x4)
     */
    inline A gradient_fx(const State& state,
                         const Control& ctrl,
                         const double step = 0.1) const override {
        double yaw = state(3);
        double v = state(2);
        double delta = ctrl(1);
        double beta_d = std::atan(std::tan(delta / 2.0));

        A dfdx = A::Zero();
        dfdx(0, 0) = 1.0;
        dfdx(0, 2) = std::cos(yaw + beta_d) * step;
        dfdx(0, 3) = -v * std::sin(yaw + beta_d) * step;

        dfdx(1, 1) = 1.0;
        dfdx(1, 2) = std::sin(yaw + beta_d) * step;
        dfdx(1, 3) = v * std::cos(yaw + beta_d) * step;

        dfdx(2, 2) = 1.0;

        dfdx(3, 2) = 2.0 * std::sin(beta_d) * step / WHEEL_BASE;
        dfdx(3, 3) = 1.0;

        return dfdx;
    }

    /**
     * @brief Compute state transition Jacobian with respect to control (B matrix)
     *
     * Computes df/du for the kinematic bicycle model.
     *
     * @param state Current state [x, y, v, yaw]
     * @param ctrl Current control [acceleration, steering]
     * @param step Time step (default 0.1)
     * @return B matrix (4x2)
     */
    inline B gradient_fu(const State& state,
                         const Control& ctrl,
                         const double step = 0.1) const override {
        double delta = ctrl(1);
        double yaw = state(3);
        double v = state(2);
        double beta_d = std::atan(std::tan(delta / 2.0));
        double beta_over_stl = 0.5 * (1.0 + std::tan(delta) * std::tan(delta)) /
                                       (1.0 + 0.25 * std::tan(delta) * std::tan(delta));

        B dfdu = B::Zero();
        dfdu(0, 1) = v * (-std::sin(beta_d + yaw)) * step * beta_over_stl;
        dfdu(1, 1) = v * std::cos(beta_d + yaw) * step * beta_over_stl;
        dfdu(2, 0) = step;
        dfdu(3, 1) = (2.0 * v * step / WHEEL_BASE) * std::cos(beta_d) * beta_over_stl;

        return dfdu;
    }

    /**
     * @brief Forward calculation (system dynamics)
     *
     * Computes the next state given current state and control.
     *
     * @param state Current state [x, y, v, yaw]
     * @param ctrl Current control [acceleration, steering]
     * @param step Time step (default 0.1)
     * @return Next state (4,)
     */
    inline State forward_calculation(const State& state,
                                     const Control& ctrl,
                                     const double step = 0.1) const override {
        double beta = std::atan(std::tan(ctrl(1)) / 2.0);

        State next_state;
        next_state(0) = state(0) + state(2) * std::cos(beta + state(3)) * step;
        next_state(1) = state(1) + state(2) * std::sin(beta + state(3)) * step;
        next_state(2) = state(2) + ctrl(0) * step;
        next_state(3) = state(3) + 2.0 * state(2) * std::sin(beta) * step / WHEEL_BASE;

        return next_state;
    }

    /**
     * @brief Get derivatives of vehicle front and rear points with respect to state
     *
     * Computes ∂(front_point)/∂state and ∂(rear_point)/∂state.
     * Each derivative is a 2x4 matrix (2D position over 4D state).
     *
     * @param yaw Heading angle (rad)
     * @return Pair of (front_point_derivative, rear_point_derivative), each 2x4
     */
    inline std::pair<Eigen::Matrix<double, 2, 4>, Eigen::Matrix<double, 2, 4>>
    get_vehicle_front_and_rear_center_derivatives(double yaw) const {
        Eigen::Matrix<double, 2, 4> front_pnt_over_state;
        Eigen::Matrix<double, 2, 4> rear_pnt_over_state;

        double half_wb = 0.5 * WHEEL_BASE;

        // Front point over state:
        // ∂front/∂x = [1, 0, 0, -half_wb*sin(yaw)]
        // ∂front/∂y = [0, 1, 0,  half_wb*cos(yaw)]
        front_pnt_over_state << 1.0, 0.0, 0.0, -half_wb * std::sin(yaw),
                                0.0, 1.0, 0.0,  half_wb * std::cos(yaw);

        // Rear point over state:
        // ∂rear/∂x = [1, 0, 0,  half_wb*sin(yaw)]
        // ∂rear/∂y = [0, 1, 0, -half_wb*cos(yaw)]
        rear_pnt_over_state << 1.0, 0.0, 0.0,  half_wb * std::sin(yaw),
                               0.0, 1.0, 0.0, -half_wb * std::cos(yaw);

        return {front_pnt_over_state, rear_pnt_over_state};
    }

    /**
     * @brief Get vehicle front and rear center positions
     *
     * Computes the 2D positions of the front and rear axles.
     *
     * @param pos Vehicle center position [x, y]
     * @param yaw Heading angle (rad)
     * @return Pair of (front_point, rear_point), each 2D
     */
    inline std::pair<Position2D, Position2D>
    get_vehicle_front_and_rear_centers(const Position2D& pos, double yaw) const {
        double half_wb = 0.5 * WHEEL_BASE;

        Position2D half_wb_vec;
        half_wb_vec(0) = half_wb * std::cos(yaw);
        half_wb_vec(1) = half_wb * std::sin(yaw);

        Position2D front_pnt = pos + half_wb_vec;
        Position2D rear_pnt = pos - half_wb_vec;

        return {front_pnt, rear_pnt};
    }

    /**
     * @brief Initialize trajectory by forward propagating dynamics
     *
     * Creates a trajectory by starting from init_state and applying controls sequentially.
     *
     * @param init_state Initial state [x, y, v, yaw]
     * @param controls Control sequence (horizon x 2)
     * @param horizon Trajectory length (default 60)
     * @return State trajectory (horizon+1 x 4)
     */
    inline States init_traj(const State& init_state, const Controls& controls, int horizon = 60) const {
        States states(horizon + 1);
        states[0] = init_state;

        for (int i = 1; i <= horizon; ++i) {
            states[i] = forward_calculation(states[i - 1], controls[i - 1]);
        }

        return states;
    }

    /**
     * @brief Get wheelbase parameter
     * @return Wheelbase in meters
     */
    inline double get_wheelbase() const { return WHEEL_BASE; }

    /**
     * @brief Get vehicle width
     * @return Width in meters
     */
    inline double get_width() const { return WIDTH; }

    /**
     * @brief Get vehicle length
     * @return Length in meters
     */
    inline double get_length() const { return LENGTH; }
};

} // namespace ilqr
