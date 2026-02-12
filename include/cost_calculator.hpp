#pragma once

#include "cost_base.hpp"
#include "state_cost.hpp"
#include "state_constraint.hpp"
#include "control_constraint.hpp"

namespace ilqr {

template<typename T = double>
class CostCalculator {
public:
    ILQR_PROBLEM_VARIABLES(T, 4, 2);

    typedef StateCost<T> StateCostType;
    typedef StateConstraint<T> StateConstraintType;
    typedef ControlConstraint<T> ControlConstraintType;

    CostCalculator(StateCostType* state_cost,
                   StateConstraintType* state_constraints,
                   ControlConstraintType* control_constraints,
                   int horizon)
        : state_cost_(state_cost),
          state_constraints_(state_constraints),
          control_constraints_(control_constraints),
          horizon_(horizon) {}

    T CalculateTotalCost(const States& states, const Controls& controls) {
        // Update reference states for current trajectory
        Matrix<T, Eigen::Dynamic, 2> positions(states.size(), 2);
        for (size_t i = 0; i < states.size(); ++i) {
            positions.row(i) << states[i](0), states[i](1);
        }
        state_cost_->get_ref_states(positions);

        // Initial state cost (step 0, zero control)
        T state_cost_val;
        Control zero_ctrl = Control::Zero();
        state_cost_->value(0, states[0], zero_ctrl, state_cost_val);

        T total_cost = state_cost_val;
        T constraint_val;
        for (int i = 1; i <= horizon_; ++i) {
            state_cost_->value(i, states[i], controls[i - 1], state_cost_val);
            total_cost += state_cost_val;

            // State constraints (use zero_ctrl for step constraints)
            state_constraints_->value(i, states[i], zero_ctrl, constraint_val);
            total_cost += constraint_val;

            // Control constraints
            control_constraints_->value(i, states[i], controls[i - 1], constraint_val);
            total_cost += constraint_val;
        }

        return total_cost;
    }

    void CalculateDerivates(const States& states, const Controls& controls,
                           VecXs& lx, MatrixCXXs& lxx, VecUs& lu, MatrixCUUs& luu, MatrixCXUs& lxu) {
        // Initialize derivative arrays
        lx.resize(horizon_ + 1);
        lxx.resize(horizon_ + 1);
        lu.resize(horizon_);
        luu.resize(horizon_);
        lxu.resize(horizon_);

        for (int i = 0; i <= horizon_; ++i) {
            lx[i] = VecX::Zero();
            lxx[i] = MatrixLXX::Zero();
        }
        for (int i = 0; i < horizon_; ++i) {
            lu[i] = VecU::Zero();
            luu[i] = MatrixLUU::Zero();
            lxu[i] = MatrixLXU::Zero();
        }

        // Step 0 derivatives (zero control)
        Control zero_ctrl = Control::Zero();
        VecX lx_step, lx_step2;
        MatrixLXX lxx_step, lxx_step2;

        state_constraints_->gradient_lx(0, states[0], zero_ctrl, lx_step);
        state_constraints_->hessian_lxx(0, states[0], zero_ctrl, lxx_step);
        lx[0] += lx_step;
        lxx[0] += lxx_step;

        state_cost_->gradient_lx(0, states[0], zero_ctrl, lx_step2);
        state_cost_->hessian_lxx(0, states[0], zero_ctrl, lxx_step2);
        lx[0] += lx_step2;
        lxx[0] += lxx_step2;

        // Loop through remaining steps
        for (int i = 1; i <= horizon_; ++i) {
            State state = states[i];
            Control control = controls[i - 1];

            // State derivatives
            state_constraints_->gradient_lx(i, state, control, lx_step);
            state_constraints_->hessian_lxx(i, state, control, lxx_step);
            lx[i] += lx_step;
            lxx[i] += lxx_step;

            state_cost_->gradient_lx(i, state, control, lx_step2);
            state_cost_->hessian_lxx(i, state, control, lxx_step2);
            lx[i] += lx_step2;
            lxx[i] += lxx_step2;

            // Control derivatives
            VecU lu_step;
            MatrixLUU luu_step;

            control_constraints_->gradient_lu(i, state, control, lu_step);
            control_constraints_->hessian_luu(i, state, control, luu_step);
            lu[i - 1] += lu_step;
            luu[i - 1] += luu_step;

            state_cost_->gradient_lu(i, state, control, lu_step);
            state_cost_->hessian_luu(i, state, control, luu_step);
            lu[i - 1] += lu_step;
            luu[i - 1] += luu_step;
        }
    }

    // Member access
    StateCostType* getStateCost() { return state_cost_; }
    StateConstraintType* getStateConstraints() { return state_constraints_; }
    ControlConstraintType* getControlConstraints() { return control_constraints_; }

private:
    StateCostType* state_cost_;
    StateConstraintType* state_constraints_;
    ControlConstraintType* control_constraints_;
    int horizon_;
};

} // namespace ilqr
