#pragma once

#include "cost_base.hpp"
#include "kinematic_model.hpp"
#include "obstacle.hpp"
#include <Eigen/Core>
#include <vector>
#include <memory>

namespace ilqr {

/**
 * @brief State constraint cost using exponential barriers
 *
 * Implements soft constraints on velocity bounds and obstacle avoidance using
 * exponential barrier functions:
 *   b = q1 * exp(q2 * constraint)
 *
 * Corresponds to Python's StateConstraint class in StateConstraint.py.
 *
 * Constraints:
 *   - Velocity upper: v <= v_max (10 m/s)
 *   - Velocity lower: v >= v_min (0 m/s)
 *   - Obstacle avoidance: ellipsoid safety margins for front and rear vehicle points
 */
class StateConstraint : public CostFunc<double, 4, 2> {
public:
    // Type definitions
    ILQR_PROBLEM_VARIABLES(double, 4, 2)

    // Constraint bounds
    static constexpr double V_MAX = 10.0;
    static constexpr double V_MIN = 0.0;

    /**
     * @brief Constructor
     *
     * @param state_dim State dimension (default 4)
     * @param control_dim Control dimension (default 2)
     * @param model Pointer to kinematic model (for vehicle geometry)
     * @param obstacles List of obstacle pointers
     */
    StateConstraint(int state_dim = 4,
                    int control_dim = 2,
                    const KinematicModel* model = nullptr,
                    const std::vector<std::shared_ptr<Obstacle>>& obstacles = {})
        : CostFunc<double, 4, 2>(state_dim, control_dim),
          model_(model),
          obstacles_(obstacles) {}

    /**
     * @brief Destructor
     */
    ~StateConstraint() override = default;

    /**
     * @brief Set the kinematic model
     * @param model Pointer to kinematic model
     */
    inline void set_model(const KinematicModel* model) { model_ = model; }

    /**
     * @brief Set obstacles list
     * @param obstacles List of obstacle pointers
     */
    inline void set_obstacles(const std::vector<std::shared_ptr<Obstacle>>& obstacles) {
        obstacles_ = obstacles;
    }

    /**
     * @brief Compute cost function value
     *
     * Sum of exponential barrier costs for velocity and obstacle constraints:
     *   L = barrier(v - v_max) + barrier(v_min - v)
     *     + Σ_obs [barrier(safety_front) + barrier(safety_rear)]
     *
     * @param step Time step
     * @param state Current state [x, y, v, yaw]
     * @param ctrl Current control [acceleration, steering]
     * @param val Output parameter for cost value
     * @return true if computation successful
     */
    inline bool value(int step,
                      const State& state,
                      const Control& ctrl,
                      double& val) const override {
        double x = state(0);
        double y = state(1);
        double v = state(2);
        double yaw = state(3);

        // Velocity constraints
        double velo_up_constraint = get_bound_constr(v, V_MAX, true);   // v - v_max
        double velo_low_constraint = get_bound_constr(v, V_MIN, false); // v_min - v

        val = exp_barrier(velo_up_constraint) + exp_barrier(velo_low_constraint);

        // Obstacle avoidance constraints
        if (model_ != nullptr) {
            Position2D pos;
            pos << x, y;

            auto [front_pnt, rear_pnt] = model_->get_vehicle_front_and_rear_centers(pos, yaw);

            for (const auto& obs : obstacles_) {
                const auto& prediction_traj = obs->get_prediction_trajectory();
                if (step < static_cast<int>(prediction_traj.size())) {
                    State obs_state = prediction_traj[step];
                    Position2D obs_center = obs_state.head<2>();

                    double front_safety = obs->ellipsoid_safety_margin(front_pnt, obs_center);
                    double rear_safety = obs->ellipsoid_safety_margin(rear_pnt, obs_center);

                    val += exp_barrier(front_safety) + exp_barrier(rear_safety);
                }
            }
        }

        return true;
    }

    /**
     * @brief Compute cost gradient with respect to state
     *
     * Chain rule for velocity and obstacle constraints:
     *   - Velocity: ∂L/∂v = q2 * barrier * ∂constraint/∂v
     *   - Obstacle: ∂L/∂x = ∂barrier/∂margin * ∂margin/∂point * ∂point/∂state
     *
     * @param step Time step
     * @param state Current state
     * @param ctrl Current control
     * @param lx Output parameter for gradient (4,)
     * @return true if computation successful
     */
    inline bool gradient_lx(int step,
                            const State& state,
                            const Control& ctrl,
                            VecX& lx) const override {
        double x = state(0);
        double y = state(1);
        double v = state(2);
        double yaw = state(3);

        lx = VecX::Zero();

        // Velocity constraints
        double velo_up_constraint = get_bound_constr(v, V_MAX, true);
        double velo_low_constraint = get_bound_constr(v, V_MIN, false);

        VecX velo_up_constraint_dx;   velo_up_constraint_dx <<   0.0, 0.0,  1.0, 0.0;
        VecX velo_low_constraint_dx;  velo_low_constraint_dx <<  0.0, 0.0, -1.0, 0.0;

        lx += exp_barrier_jacobian(velo_up_constraint, velo_up_constraint_dx);
        lx += exp_barrier_jacobian(velo_low_constraint, velo_low_constraint_dx);

        // Obstacle avoidance constraints
        if (model_ != nullptr) {
            Position2D pos;
            pos << x, y;

            auto [front_pnt, rear_pnt] = model_->get_vehicle_front_and_rear_centers(pos, yaw);

            for (const auto& obs : obstacles_) {
                const auto& prediction_traj = obs->get_prediction_trajectory();
                if (step < static_cast<int>(prediction_traj.size())) {
                    State obs_state = prediction_traj[step];
                    Position2D obs_center = obs_state.head<2>();

                    // Safety margins
                    double front = obs->ellipsoid_safety_margin(front_pnt, obs_center);
                    double rear = obs->ellipsoid_safety_margin(rear_pnt, obs_center);

                    // Safety margin derivatives w.r.t. vehicle points (2D)
                    Position2D front_safety_margin_over_ego_front =
                        obs->ellipsoid_safety_margin_derivatives(front_pnt, obs_center);
                    Position2D rear_safety_margin_over_ego_rear =
                        obs->ellipsoid_safety_margin_derivatives(rear_pnt, obs_center);

                    // Vehicle point derivatives w.r.t. state (2x4 matrices)
                    auto [ego_front_over_state, ego_rear_over_state] =
                        model_->get_vehicle_front_and_rear_center_derivatives(yaw);

                    // Chain rule: ∂margin/∂state = ∂margin/∂point * ∂point/∂state
                    VecX front_safety_margin_over_state =
                        front_safety_margin_over_ego_front.transpose() * ego_front_over_state;
                    VecX rear_safety_margin_over_state =
                        rear_safety_margin_over_ego_rear.transpose() * ego_rear_over_state;

                    // Add to gradient
                    lx += exp_barrier_jacobian(front, front_safety_margin_over_state);
                    lx += exp_barrier_jacobian(rear, rear_safety_margin_over_state);
                }
            }
        }

        return true;
    }

    /**
     * @brief Compute cost gradient with respect to control
     *
     * State constraints don't depend on control, so gradient is zero.
     *
     * @param step Time step
     * @param state Current state
     * @param ctrl Current control
     * @param lu Output parameter for gradient (2,)
     * @return true if computation successful
     */
    inline bool gradient_lu(int step,
                            const State& state,
                            const Control& ctrl,
                            VecU& lu) const override {
        lu = VecU::Zero();
        return true;
    }

    /**
     * @brief Compute cost Hessian with respect to state
     *
     * Chain rule for velocity and obstacle constraints:
     *   ∂²L/∂x² = q2² * barrier * (∂c/∂x) * (∂c/∂x).T
     *
     * @param step Time step
     * @param state Current state
     * @param ctrl Current control
     * @param lxx Output parameter for Hessian (4x4)
     * @return true if computation successful
     */
    inline bool hessian_lxx(int step,
                            const State& state,
                            const Control& ctrl,
                            MatrixLXX& lxx) const override {
        double x = state(0);
        double y = state(1);
        double v = state(2);
        double yaw = state(3);

        lxx = MatrixLXX::Zero();

        // Velocity constraints
        double velo_up_constraint = get_bound_constr(v, V_MAX, true);
        double velo_low_constraint = get_bound_constr(v, V_MIN, false);

        VecX velo_up_constraint_dx;   velo_up_constraint_dx <<   0.0, 0.0,  1.0, 0.0;
        VecX velo_low_constraint_dx;  velo_low_constraint_dx <<  0.0, 0.0, -1.0, 0.0;

        lxx += exp_barrier_hessian(velo_up_constraint, velo_up_constraint_dx);
        lxx += exp_barrier_hessian(velo_low_constraint, velo_low_constraint_dx);

        // Obstacle avoidance constraints
        if (model_ != nullptr) {
            Position2D pos;
            pos << x, y;

            auto [front_pnt, rear_pnt] = model_->get_vehicle_front_and_rear_centers(pos, yaw);

            for (const auto& obs : obstacles_) {
                const auto& prediction_traj = obs->get_prediction_trajectory();
                if (step < static_cast<int>(prediction_traj.size())) {
                    State obs_state = prediction_traj[step];
                    Position2D obs_center = obs_state.head<2>();

                    // Safety margins
                    double front = obs->ellipsoid_safety_margin(front_pnt, obs_center);
                    double rear = obs->ellipsoid_safety_margin(rear_pnt, obs_center);

                    // Safety margin derivatives w.r.t. vehicle points (2D)
                    Position2D front_safety_margin_over_ego_front =
                        obs->ellipsoid_safety_margin_derivatives(front_pnt, obs_center);
                    Position2D rear_safety_margin_over_ego_rear =
                        obs->ellipsoid_safety_margin_derivatives(rear_pnt, obs_center);

                    // Vehicle point derivatives w.r.t. state (2x4 matrices)
                    auto [ego_front_over_state, ego_rear_over_state] =
                        model_->get_vehicle_front_and_rear_center_derivatives(yaw);

                    // Chain rule: ∂margin/∂state = ∂margin/∂point * ∂point/∂state
                    VecX front_safety_margin_over_state =
                        front_safety_margin_over_ego_front.transpose() * ego_front_over_state;
                    VecX rear_safety_margin_over_state =
                        rear_safety_margin_over_ego_rear.transpose() * ego_rear_over_state;

                    // Add to Hessian
                    lxx += exp_barrier_hessian(front, front_safety_margin_over_state);
                    lxx += exp_barrier_hessian(rear, rear_safety_margin_over_state);
                }
            }
        }

        return true;
    }

    /**
     * @brief Compute cost Hessian with respect to control
     *
     * State constraints don't depend on control, so Hessian is zero.
     *
     * @param step Time step
     * @param state Current state
     * @param ctrl Current control
     * @param luu Output parameter for Hessian (2x2)
     * @return true if computation successful
     */
    inline bool hessian_luu(int step,
                            const State& state,
                            const Control& ctrl,
                            MatrixLUU& luu) const override {
        luu = MatrixLUU::Zero();
        return true;
    }

    /**
     * @brief Compute mixed cost Hessian (state x control)
     *
     * State constraints don't depend on control, so mixed Hessian is zero.
     *
     * @param step Time step
     * @param state Current state
     * @param ctrl Current control
     * @param lxu Output parameter for Hessian (4x2)
     * @return true if computation successful
     */
    inline bool hessian_lxu(int step,
                            const State& state,
                            const Control& ctrl,
                            MatrixLXU& lxu) const override {
        lxu = MatrixLXU::Zero();
        return true;
    }

    /**
     * @brief Get velocity maximum bound
     * @return v_max in m/s
     */
    static constexpr double get_v_max() { return V_MAX; }

    /**
     * @brief Get velocity minimum bound
     * @return v_min in m/s
     */
    static constexpr double get_v_min() { return V_MIN; }

private:
    const KinematicModel* model_;                                      ///< Kinematic model pointer
    std::vector<std::shared_ptr<Obstacle>> obstacles_;                 ///< Obstacle list
};

} // namespace ilqr
