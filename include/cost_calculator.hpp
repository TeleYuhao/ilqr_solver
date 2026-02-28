/**
 * @file cost_calculator.hpp
 * @brief Aggregates all cost components for iLQR trajectory optimization
 *
 * Combines StateCost, StateConstraint, and ControlConstraint into a single
 * cost function interface for the iLQR solver.
 */

#ifndef ILQR_COST_CALCULATOR_HPP
#define ILQR_COST_CALCULATOR_HPP

#include "common_variable.hpp"
#include "state_cost.hpp"
#include "state_constraint.hpp"
#include "control_constraint.hpp"
#include "kinematic_model.hpp"

#include <vector>
#include <memory>
#include <iostream>

namespace ilqr {

/**
 * @brief Aggregates all cost components for iLQR optimization
 *
 * This class combines:
 * - StateCost: Quadratic tracking cost to reference trajectory
 * - StateConstraint: Velocity bounds and obstacle avoidance
 * - ControlConstraint: Acceleration and steering limits
 */
class CostCalculator {
public:
    // Type definitions
    ILQR_PROBLEM_VARIABLES(double, 4, 2)

    // Dimension constants
    static constexpr int STATE_DIM = 4;
    static constexpr int CONTROL_DIM = 2;

    /**
     * @brief Constructor with individual cost components
     *
     * @param state_cost Shared pointer to StateCost
     * @param state_constraint Shared pointer to StateConstraint
     * @param control_constraint Shared pointer to ControlConstraint
     * @param horizon Planning horizon (default: 60)
     */
    CostCalculator(std::shared_ptr<StateCost> state_cost,
                   std::shared_ptr<StateConstraint> state_constraint,
                   std::shared_ptr<ControlConstraint> control_constraint,
                   int horizon = 60)
        : state_cost_(state_cost)
        , state_constraint_(state_constraint)
        , control_constraint_(control_constraint)
        , horizon_(horizon)
        , state_dim_(STATE_DIM)
        , control_dim_(CONTROL_DIM)
    {}

    /**
     * @brief Default constructor - creates cost components internally
     *
     * @param model Kinematic model pointer (for StateConstraint)
     * @param obstacles Vector of obstacle pointers
     * @param ref_waypoints Reference trajectory waypoints (2 x N matrix)
     * @param horizon Planning horizon
     */
    CostCalculator(KinematicModel* model,
                   const std::vector<std::shared_ptr<Obstacle>>& obstacles,
                   const Eigen::Matrix<double, 2, Eigen::Dynamic>& ref_waypoints,
                   int horizon = 60)
        : horizon_(horizon)
        , state_dim_(STATE_DIM)
        , control_dim_(CONTROL_DIM)
    {
        // Create StateCost with Q and R matrices matching Python test.py
        // Python: Q = np.diag([1.0, 1.0, 0.5, 0])
        // Python: R = np.diag([1.0, 1.0])
        MatrixLXX Q = MatrixLXX::Zero();
        Q(0, 0) = 1.0;   // x position weight
        Q(1, 1) = 1.0;   // y position weight
        Q(2, 2) = 0.5;   // velocity weight
        Q(3, 3) = 0.0;   // yaw weight (no cost)

        MatrixLUU R = MatrixLUU::Identity();
        state_cost_ = std::make_shared<StateCost>(Q, R, ref_waypoints);

        // Create StateConstraint
        state_constraint_ = std::make_shared<StateConstraint>(state_dim_, control_dim_, model, obstacles);

        // Create ControlConstraint
        control_constraint_ = std::make_shared<ControlConstraint>();
    }

    /**
     * @brief Calculate total cost across the entire trajectory
     *
     * Cost = state_cost + state_constraint + control_constraint
     * summed over all timesteps.
     *
     * @param states State trajectory (horizon+1 x 4)
     * @param controls Control trajectory (horizon x 2)
     * @return Total cost
     */
    double CalculateTotalCost(const States& states, const Controls& controls) {
        // Update reference states from trajectory positions
        Eigen::Matrix<double, Eigen::Dynamic, 2> positions(states.size(), 2);
        for (size_t i = 0; i < states.size(); ++i) {
            positions(i, 0) = states[i](0);
            positions(i, 1) = states[i](1);
        }
        state_cost_->get_ref_states(positions);

        // Initial state cost (no control at t=0)
        double val, total_cost;
        state_cost_->value(0, states[0], Control::Zero(), total_cost);

        // Sum costs over horizon
        for (int i = 1; i <= horizon_; ++i) {
            // State cost
            state_cost_->value(i, states[i], controls[i-1], val);
            total_cost += val;

            // Constraints: state + control
            state_constraint_->value(i, states[i], controls[i-1], val);
            total_cost += val;
            control_constraint_->value(i, states[i], controls[i-1], val);
            total_cost += val;
        }

        return total_cost;
    }

    /**
     * @brief Calculate all cost derivatives across the trajectory
     *
     * Aggregates derivatives from all cost components:
     * - lx, lxx: State cost gradients
     * - lu, luu: Control cost gradients
     * - lxu: Cross derivatives (currently zeros)
     *
     * @param states State trajectory (horizon+1 x 4)
     * @param controls Control trajectory (horizon x 2)
     * @param lx Output: state gradient (horizon+1 x 4)
     * @param lxx Output: state Hessian (horizon+1 x 4x4)
     * @param lu Output: control gradient (horizon x 2)
     * @param luu Output: control Hessian (horizon x 2x2)
     * @param lxu Output: cross Hessian (horizon x 4x2)
     */
    void CalculateDerivatives(const States& states,
                              const Controls& controls,
                              VecXs& lx,
                              MatrixCXXs& lxx,
                              VecUs& lu,
                              MatrixCUUs& luu,
                              MatrixCXUs& lxu) {
        // Update reference states from trajectory positions
        Eigen::Matrix<double, Eigen::Dynamic, 2> positions(states.size(), 2);
        for (size_t i = 0; i < states.size(); ++i) {
            positions(i, 0) = states[i](0);
            positions(i, 1) = states[i](1);
        }
        state_cost_->get_ref_states(positions);

        // Resize outputs
        lx.resize(horizon_ + 1);
        lxx.resize(horizon_ + 1);
        lu.resize(horizon_);
        luu.resize(horizon_);
        lxu.resize(horizon_);

        // Initialize to zero
        for (int i = 0; i <= horizon_; ++i) {
            lx[i] = VecX::Zero();
            lxx[i] = MatrixLXX::Zero();
        }
        for (int i = 0; i < horizon_; ++i) {
            lu[i] = VecU::Zero();
            luu[i] = MatrixLUU::Zero();
            lxu[i] = MatrixLXU::Zero();
        }

        // Initial timestep (t=0): only state contributions (no control yet)
        VecX lx_sc, lx_state;
        MatrixLXX lxx_sc, lxx_state;

        state_constraint_->gradient_lx(0, states[0], Control::Zero(), lx_sc);
        state_constraint_->hessian_lxx(0, states[0], Control::Zero(), lxx_sc);

        state_cost_->gradient_lx(0, states[0], Control::Zero(), lx_state);
        state_cost_->hessian_lxx(0, states[0], Control::Zero(), lxx_state);

        lx[0] += lx_sc + lx_state;
        lxx[0] += lxx_sc + lxx_state;

        // Sum contributions for t=1 to horizon
        for (int i = 1; i <= horizon_; ++i) {
            VecX lx_i, lx_sc_i;
            MatrixLXX lxx_i, lxx_sc_i;
            VecU lu_i, lu_sc_i;
            MatrixLUU luu_i, luu_sc_i;
            MatrixLXU lxu_i, lxu_sc_i;

            // State constraint derivatives
            state_constraint_->gradient_lx(i, states[i], controls[i-1], lx_sc_i);
            state_constraint_->hessian_lxx(i, states[i], controls[i-1], lxx_sc_i);

            // State cost derivatives
            state_cost_->gradient_lx(i, states[i], controls[i-1], lx_i);
            state_cost_->hessian_lxx(i, states[i], controls[i-1], lxx_i);
            state_cost_->gradient_lu(i, states[i], controls[i-1], lu_i);
            state_cost_->hessian_luu(i, states[i], controls[i-1], luu_i);
            state_cost_->hessian_lxu(i, states[i], controls[i-1], lxu_i);

            // Control constraint derivatives
            control_constraint_->gradient_lu(i, states[i], controls[i-1], lu_sc_i);
            control_constraint_->hessian_luu(i, states[i], controls[i-1], luu_sc_i);
            control_constraint_->hessian_lxu(i, states[i], controls[i-1], lxu_sc_i);

            // Accumulate
            lx[i] += lx_sc_i + lx_i;
            lxx[i] += lxx_sc_i + lxx_i;
            lu[i-1] += lu_sc_i + lu_i;
            luu[i-1] += luu_sc_i + luu_i;
            lxu[i-1] += lxu_sc_i + lxu_i;
        }
    }

    /**
     * @brief Get reference states for tracking cost
     * @param positions 2D positions along trajectory
     */
    void get_ref_states(const Eigen::Matrix<double, Eigen::Dynamic, 2>& positions) {
        state_cost_->get_ref_states(positions);
    }

    /**
     * @brief Get horizon length
     */
    int get_horizon() const { return horizon_; }

    /**
     * @brief Get state dimension
     */
    int get_state_dim() const { return state_dim_; }

    /**
     * @brief Get control dimension
     */
    int get_control_dim() const { return control_dim_; }

    // Accessors for individual cost components
    std::shared_ptr<StateCost> get_state_cost() { return state_cost_; }
    std::shared_ptr<StateConstraint> get_state_constraint() { return state_constraint_; }
    std::shared_ptr<ControlConstraint> get_control_constraint() { return control_constraint_; }

private:
    std::shared_ptr<StateCost> state_cost_;
    std::shared_ptr<StateConstraint> state_constraint_;
    std::shared_ptr<ControlConstraint> control_constraint_;

    int horizon_;
    int state_dim_;
    int control_dim_;
};

} // namespace ilqr

#endif // ILQR_COST_CALCULATOR_HPP
