#pragma once

#include "common_variable.hpp"
#include <Eigen/Core>
#include <vector>
#include <cmath>

namespace ilqr {

/**
 * @brief Ellipsoid obstacle with constant velocity prediction
 *
 * Represents an obstacle as an ellipsoid with safety margin.
 * Corresponds to Python's obstacle class in obstacle_base.py.
 *
 * The obstacle state is [x, y, v, yaw] where:
 *   - x, y: position (m)
 *   - v: velocity (m/s)
 *   - yaw: heading angle (rad)
 *
 * The obstacle attributes are [width, length, safety_buffer] where:
 *   - width: obstacle width (m)
 *   - length: obstacle length (m)
 *   - safety_buffer: additional safety margin (m)
 */
class Obstacle {
public:
    // Type definitions for 2D geometry
    typedef Eigen::Matrix<double, 2, 1> Position2D;
    typedef Eigen::Matrix<double, 2, 2> RotationMatrix;
    typedef Eigen::Matrix<double, 4, 1> State4D;
    typedef Eigen::Matrix<double, 3, 1> Attributes;

    // Ego vehicle parameters
    static constexpr double EGO_WIDTH = 2.0;
    static constexpr double DEFAULT_DT = 0.1;

    /**
     * @brief Constructor
     *
     * @param state Obstacle state [x, y, v, yaw]
     * @param attr Obstacle attributes [width, length, safety_buffer]
     */
    Obstacle(const State4D& state, const Attributes& attr)
        : state_(state),
          attr_(attr),
          pos_(state.head<2>()),
          yaw_(state(3)),
          ego_pnt_radius_(EGO_WIDTH / 2.0) {

        obs_width_ = attr_(0);
        obs_length_ = attr_(1);
        d_safe_ = attr_(2);

        get_ellipsoid_obstacle_scales();
        prediction_traj_ = const_velo_prediction(state_, 60);
    }

    /**
     * @brief Destructor
     */
    ~Obstacle() = default;

    /**
     * @brief Compute ellipsoid safety margin
     *
     * Computes the constraint value: 1 - (x_std²/a² + y_std²/b²)
     * where (x_std, y_std) is the point rotated to obstacle frame.
     * Positive value means inside safe region (constraint violated).
     *
     * @param pnt Query point in global frame [x, y]
     * @param elp_center Ellipsoid center in global frame [x, y]
     * @return Safety margin (positive = violated, negative = safe)
     */
    inline double ellipsoid_safety_margin(const Position2D& pnt,
                                          const Position2D& elp_center) const {
        // Compute difference vector
        Position2D diff = pnt - elp_center;

        // Rotation matrix to transform to obstacle frame
        RotationMatrix rotation_matrix;
        double cos_theta = std::cos(yaw_);
        double sin_theta = std::sin(yaw_);
        rotation_matrix << cos_theta, -sin_theta,
                           sin_theta,  cos_theta;

        // Transform point to obstacle frame
        Position2D pnt_std = rotation_matrix.transpose() * diff;

        // Compute ellipsoid constraint: 1 - (x²/a² + y²/b²)
        double result = 1.0 - ((pnt_std(0) * pnt_std(0)) / (a_ * a_) +
                               (pnt_std(1) * pnt_std(1)) / (b_ * b_));

        return result;
    }

    /**
     * @brief Compute ellipsoid safety margin gradient
     *
     * Computes ∂(safety_margin)/∂pnt using chain rule.
     *
     * Chain rule breakdown:
     *   (1) constraint over standard point: ∂c/∂p_std
     *   (2) standard point over difference: ∂p_std/∂diff
     *   (3) difference over original point: ∂diff/∂pnt
     *
     * @param pnt Query point in global frame [x, y]
     * @param elp_center Ellipsoid center in global frame [x, y]
     * @return Gradient [∂c/∂x, ∂c/∂y]
     */
    inline Position2D ellipsoid_safety_margin_derivatives(const Position2D& pnt,
                                                           const Position2D& elp_center) const {
        // Compute difference vector
        Position2D diff = pnt - elp_center;

        // Rotation matrix to transform to obstacle frame
        RotationMatrix rotation_matrix;
        double cos_theta = std::cos(yaw_);
        double sin_theta = std::sin(yaw_);
        rotation_matrix << cos_theta, -sin_theta,
                           sin_theta,  cos_theta;

        // Transform point to obstacle frame
        Position2D pnt_std = rotation_matrix.transpose() * diff;

        // (1) constraint over standard point vector
        Position2D res_over_pnt_std;
        res_over_pnt_std(0) = -2.0 * pnt_std(0) / (a_ * a_);
        res_over_pnt_std(1) = -2.0 * pnt_std(1) / (b_ * b_);

        // (2) standard point over difference (transpose of rotation matrix)
        RotationMatrix pnt_std_over_diff = rotation_matrix;

        // (3) difference over original point (identity matrix)
        RotationMatrix diff_over_pnt = RotationMatrix::Identity();

        // Chain (1)(2)(3) together: ∂c/∂pnt = ∂c/∂p_std * ∂p_std/∂diff * ∂diff/∂pnt
        Position2D res_over_pnt = res_over_pnt_std.transpose() * pnt_std_over_diff * diff_over_pnt;

        return res_over_pnt;
    }

    /**
     * @brief Single-step kinematic propagation
     *
     * Propagates obstacle state by one time step with constant velocity model.
     *
     * @param x0 Current state [x, y, v, yaw]
     * @param dt Time step (default 0.1)
     * @return Next state [x, y, v, yaw]
     */
    static inline State4D kinematic_propagate(const State4D& x0, double dt = DEFAULT_DT) {
        double x = x0(0);
        double y = x0(1);
        double v = x0(2);
        double yaw = x0(3);

        State4D next_state;
        next_state(0) = x + v * std::cos(yaw) * dt;
        next_state(1) = y + v * std::sin(yaw) * dt;
        next_state(2) = v;  // constant velocity
        next_state(3) = yaw;  // constant heading

        return next_state;
    }

    /**
     * @brief Constant velocity prediction
     *
     * Predicts obstacle trajectory for specified number of steps using constant velocity model.
     *
     * @param x0 Initial state [x, y, v, yaw]
     * @param steps Number of prediction steps
     * @param dt Time step (default 0.1)
     * @return Predicted states (steps+1 x 4)
     */
    static inline std::vector<State4D, Eigen::aligned_allocator<State4D>>
    const_velo_prediction(const State4D& x0, int steps, double dt = DEFAULT_DT) {
        std::vector<State4D, Eigen::aligned_allocator<State4D>> predicted_states;
        predicted_states.reserve(steps + 1);
        predicted_states.push_back(x0);

        State4D cur_x = x0;
        for (int i = 0; i < steps; ++i) {
            State4D next_x = kinematic_propagate(cur_x, dt);
            cur_x = next_x;
            predicted_states.push_back(next_x);
        }

        return predicted_states;
    }

    /**
     * @brief Get ellipsoid scale a (semi-major axis along x)
     * @return Scale a in meters
     */
    inline double get_a() const { return a_; }

    /**
     * @brief Get ellipsoid scale b (semi-minor axis along y)
     * @return Scale b in meters
     */
    inline double get_b() const { return b_; }

    /**
     * @brief Get obstacle position
     * @return Position [x, y]
     */
    inline const Position2D& get_position() const { return pos_; }

    /**
     * @brief Get obstacle yaw
     * @return Yaw angle in radians
     */
    inline double get_yaw() const { return yaw_; }

    /**
     * @brief Get obstacle state
     * @return Full state [x, y, v, yaw]
     */
    inline const State4D& get_state() const { return state_; }

    /**
     * @brief Get prediction trajectory
     * @return Vector of predicted states
     */
    inline const std::vector<State4D, Eigen::aligned_allocator<State4D>>&
    get_prediction_trajectory() const { return prediction_traj_; }

private:
    /**
     * @brief Compute ellipsoid scales from obstacle attributes
     *
     * Sets:
     *   a = 0.5 * obs_length + d_safe + ego_pnt_radius
     *   b = 0.5 * obs_width + d_safe + ego_pnt_radius
     */
    inline void get_ellipsoid_obstacle_scales() {
        a_ = 0.5 * obs_length_ + d_safe_ + ego_pnt_radius_;
        b_ = 0.5 * obs_width_ + d_safe_ + ego_pnt_radius_;
    }

    State4D state_;                              ///< Obstacle state [x, y, v, yaw]
    Attributes attr_;                            ///< Obstacle attributes [width, length, safety]
    Position2D pos_;                             ///< Obstacle position [x, y]
    double yaw_;                                 ///< Obstacle heading angle (rad)
    double ego_pnt_radius_;                      ///< Ego vehicle half-width (m)
    double obs_width_;                           ///< Obstacle width (m)
    double obs_length_;                          ///< Obstacle length (m)
    double d_safe_;                              ///< Safety buffer (m)
    double a_;                                   ///< Ellipsoid semi-major axis (m)
    double b_;                                   ///< Ellipsoid semi-minor axis (m)
    std::vector<State4D, Eigen::aligned_allocator<State4D>> prediction_traj_;  ///< Predicted trajectory
};

} // namespace ilqr
