#pragma once

#include "cost_base.hpp"
#include <Eigen/Core>
#include <cmath>

namespace ilqr {

/**
 * @brief Quadratic tracking cost function
 *
 * Computes quadratic cost for tracking reference trajectory:
 *   L = (x - x_ref)^T Q (x - x_ref) + u^T R u
 *
 * Corresponds to Python's StateCost class in state_cost.py.
 *
 * The reference state includes:
 *   - x, y: position reference from waypoints
 *   - v: reference velocity (default 6.0 m/s)
 *   - yaw: reference heading (default 0)
 */
class StateCost : public CostFunc<double, 4, 2> {
public:
    // Type definitions
    ILQR_PROBLEM_VARIABLES(double, 4, 2)

    // Default parameters
    static constexpr double DEFAULT_REF_VELO = 6.0;
    static constexpr int DEFAULT_HORIZON = 60;

    /**
     * @brief Constructor
     *
     * @param Q State weight matrix (4x4)
     * @param R Control weight matrix (2x2)
     * @param ref_waypoints Reference waypoints (2 x N matrix, [longitudinal; lateral])
     * @param state_dim State dimension (default 4)
     * @param control_dim Control dimension (default 2)
     */
    StateCost(const MatrixLXX& Q,
              const MatrixLUU& R,
              const Eigen::Matrix<double, 2, Eigen::Dynamic>& ref_waypoints,
              int state_dim = 4,
              int control_dim = 2)
        : CostFunc<double, 4, 2>(state_dim, control_dim),
          Q_(Q),
          R_(R),
          ref_waypoints_(ref_waypoints),
          ref_velo_(DEFAULT_REF_VELO),
          horizon_(DEFAULT_HORIZON) {

        // Initialize ref_states_ to empty, will be populated by get_ref_states()
        ref_states_initialized_ = false;
    }

    /**
     * @brief Destructor
     */
    ~StateCost() override = default;

    /**
     * @brief Get reference states from positions
     *
     * Finds the closest reference waypoint for each position and constructs
     * the full reference state trajectory.
     *
     * @param positions Position trajectory (N x 2 matrix, [x, y])
     */
    inline void get_ref_states(const Eigen::Matrix<double, Eigen::Dynamic, 2>& positions) {
        int n_steps = positions.rows();

        // Reshape waypoints for broadcasting: (N_waypoints x 2 x 1)
        int n_waypoints = ref_waypoints_.cols();
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ref_waypoints_transposed =
            ref_waypoints_.transpose();  // (N_waypoints x 2)

        // Compute distances: for each position, find distance to all waypoints
        // distances shape: (N_waypoints x N_steps)
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> distances(n_waypoints, n_steps);

        for (int i = 0; i < n_steps; ++i) {
            for (int j = 0; j < n_waypoints; ++j) {
                double dx = positions(i, 0) - ref_waypoints_(0, j);
                double dy = positions(i, 1) - ref_waypoints_(1, j);
                distances(j, i) = dx * dx + dy * dy;
            }
        }

        // Find index of closest waypoint for each position
        Eigen::VectorXi arg_min_dist_indices(n_steps);
        for (int i = 0; i < n_steps; ++i) {
            Eigen::Index min_idx;
            distances.col(i).minCoeff(&min_idx);
            arg_min_dist_indices(i) = static_cast<int>(min_idx);
        }

        // Construct ref_states: (N_steps+1 x 4)
        ref_states_.resize(n_steps + 1, 4);

        for (int i = 0; i <= n_steps; ++i) {
            // Use last index if i == n_steps
            int idx = (i < n_steps) ? arg_min_dist_indices(i) : arg_min_dist_indices(n_steps - 1);

            ref_states_(i, 0) = ref_waypoints_(0, idx);  // x reference
            ref_states_(i, 1) = ref_waypoints_(1, idx);  // y reference
            ref_states_(i, 2) = ref_velo_;               // velocity reference
            ref_states_(i, 3) = 0.0;                     // yaw reference
        }

        ref_states_initialized_ = true;
    }

    /**
     * @brief Compute cost function value
     *
     * L = (x - x_ref)^T Q (x - x_ref) + u^T R u
     *
     * @param step Time step
     * @param state Current state
     * @param ctrl Current control
     * @param val Output parameter for cost value
     * @return true if computation successful
     */
    inline bool value(int step,
                      const State& state,
                      const Control& ctrl,
                      double& val) const override {
        if (!ref_states_initialized_) {
            return false;
        }

        // Get reference state for this step
        VecX ref_state = ref_states_.row(step);

        // State difference
        VecX state_diff = state - ref_state;

        // Quadratic cost: state_diff.T @ Q @ state_diff + ctrl.T @ R @ ctrl
        double state_cost = state_diff.transpose() * Q_ * state_diff;
        double control_cost = ctrl.transpose() * R_ * ctrl;

        val = state_cost + control_cost;
        return true;
    }

    /**
     * @brief Compute cost gradient with respect to state
     *
     * ∂L/∂x = 2 * Q * (x - x_ref)
     *
     * @param step Time step
     * @param state Current state
     * @param ctrl Current control
     * @param lx Output parameter for gradient
     * @return true if computation successful
     */
    inline bool gradient_lx(int step,
                            const State& state,
                            const Control& ctrl,
                            VecX& lx) const override {
        if (!ref_states_initialized_) {
            return false;
        }

        // Get reference state for this step
        VecX ref_state = ref_states_.row(step);

        // State difference
        VecX state_diff = state - ref_state;

        // Gradient: 2 * Q @ state_diff
        lx = 2.0 * (Q_ * state_diff);
        return true;
    }

    /**
     * @brief Compute cost gradient with respect to control
     *
     * ∂L/∂u = 2 * R * u
     *
     * @param step Time step
     * @param state Current state
     * @param ctrl Current control
     * @param lu Output parameter for gradient
     * @return true if computation successful
     */
    inline bool gradient_lu(int step,
                            const State& state,
                            const Control& ctrl,
                            VecU& lu) const override {
        // Gradient: 2 * R @ ctrl
        lu = 2.0 * (R_ * ctrl);
        return true;
    }

    /**
     * @brief Compute cost Hessian with respect to state
     *
     * ∂²L/∂x² = 2 * Q
     *
     * @param step Time step
     * @param state Current state
     * @param ctrl Current control
     * @param lxx Output parameter for Hessian
     * @return true if computation successful
     */
    inline bool hessian_lxx(int step,
                            const State& state,
                            const Control& ctrl,
                            MatrixLXX& lxx) const override {
        // Hessian: 2 * Q
        lxx = 2.0 * Q_;
        return true;
    }

    /**
     * @brief Compute cost Hessian with respect to control
     *
     * ∂²L/∂u² = 2 * R
     *
     * @param step Time step
     * @param state Current state
     * @param ctrl Current control
     * @param luu Output parameter for Hessian
     * @return true if computation successful
     */
    inline bool hessian_luu(int step,
                            const State& state,
                            const Control& ctrl,
                            MatrixLUU& luu) const override {
        // Hessian: 2 * R
        luu = 2.0 * R_;
        return true;
    }

    /**
     * @brief Compute mixed cost Hessian (state x control)
     *
     * ∂²L/∂x∂u = 0 (no cross terms in quadratic cost)
     *
     * @param step Time step
     * @param state Current state
     * @param ctrl Current control
     * @param lxu Output parameter for Hessian
     * @return true if computation successful
     */
    inline bool hessian_lxu(int step,
                            const State& state,
                            const Control& ctrl,
                            MatrixLXU& lxu) const override {
        // Mixed Hessian: zeros
        lxu = MatrixLXU::Zero();
        return true;
    }

    /**
     * @brief Set reference velocity
     * @param ref_velo Reference velocity (m/s)
     */
    inline void set_ref_velo(double ref_velo) { ref_velo_ = ref_velo; }

    /**
     * @brief Get reference velocity
     * @return Reference velocity (m/s)
     */
    inline double get_ref_velo() const { return ref_velo_; }

    /**
     * @brief Get Q matrix
     * @return State weight matrix
     */
    inline const MatrixLXX& get_Q() const { return Q_; }

    /**
     * @brief Get R matrix
     * @return Control weight matrix
     */
    inline const MatrixLUU& get_R() const { return R_; }

    /**
     * @brief Check if reference states are initialized
     * @return true if get_ref_states() has been called
     */
    inline bool is_ref_states_initialized() const { return ref_states_initialized_; }

private:
    MatrixLXX Q_;                                            ///< State weight matrix
    MatrixLUU R_;                                            ///< Control weight matrix
    Eigen::Matrix<double, 2, Eigen::Dynamic> ref_waypoints_; ///< Reference waypoints (2 x N)
    double ref_velo_;                                        ///< Reference velocity (m/s)
    int horizon_;                                            ///< Prediction horizon
    Eigen::Matrix<double, Eigen::Dynamic, 4> ref_states_;    ///< Reference state trajectory
    bool ref_states_initialized_;                            ///< Whether ref_states_ is initialized
};

} // namespace ilqr
