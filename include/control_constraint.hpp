#pragma once

#include "cost_base.hpp"
#include <Eigen/Core>
#include <cmath>

namespace ilqr {

/**
 * @brief Control constraint cost using exponential barriers
 *
 * Implements soft constraints on acceleration and steering limits using
 * exponential barrier functions:
 *   b = q1 * exp(q2 * constraint)
 *
 * Corresponds to Python's ControlConstraint class in ControlConstraint.py.
 *
 * Constraints:
 *   - Acceleration upper: a <= a_max
 *   - Acceleration lower: a >= a_min
 *   - Steering upper: delta <= delta_max
 *   - Steering lower: delta >= delta_min
 */
class ControlConstraint : public CostFunc<double, 4, 2> {
public:
    // Type definitions
    ILQR_PROBLEM_VARIABLES(double, 4, 2)

    // Constraint bounds
    static constexpr double V_MAX = 10.0;
    static constexpr double V_MIN = 0.0;
    static constexpr double A_MAX = 2.0;
    static constexpr double A_MIN = -2.0;
    static constexpr double DELTA_MAX = 1.57;
    static constexpr double DELTA_MIN = -1.57;

    // Barrier parameters (from cost_base.py)
    static constexpr double Q1 = 5.5;
    static constexpr double Q2 = 5.75;

    /**
     * @brief Constructor
     *
     * @param state_dim State dimension (default 4)
     * @param control_dim Control dimension (default 2)
     */
    ControlConstraint(int state_dim = 4, int control_dim = 2)
        : CostFunc<double, 4, 2>(state_dim, control_dim) {}

    /**
     * @brief Destructor
     */
    ~ControlConstraint() override = default;

    /**
     * @brief Compute cost function value
     *
     * Sum of exponential barrier costs for all 4 constraints:
     *   L = barrier(a - a_max) + barrier(a_min - a)
     *     + barrier(delta - delta_max) + barrier(delta_min - delta)
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
        double a = ctrl(0);      // acceleration
        double delta = ctrl(1);  // steering angle

        // Acceleration constraints
        double acc_up_constraint = get_bound_constr(a, A_MAX, true);   // a - a_max
        double acc_low_constraint = get_bound_constr(a, A_MIN, false);  // a_min - a

        // Steering constraints
        double delta_up_constraint = get_bound_constr(delta, DELTA_MAX, true);   // delta - delta_max
        double delta_low_constraint = get_bound_constr(delta, DELTA_MIN, false); // delta_min - delta

        // Sum of barrier costs
        val = exp_barrier(acc_up_constraint) +
              exp_barrier(acc_low_constraint) +
              exp_barrier(delta_up_constraint) +
              exp_barrier(delta_low_constraint);

        return true;
    }

    /**
     * @brief Compute cost gradient with respect to state
     *
     * Control constraints don't depend on state, so gradient is zero.
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
        lx = VecX::Zero();
        return true;
    }

    /**
     * @brief Compute cost gradient with respect to control
     *
     * Applies chain rule through barrier functions:
     *   ∂L/∂u = Σ ∂barrier/∂constraint * ∂constraint/∂u
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
        double a = ctrl(0);      // acceleration
        double delta = ctrl(1);  // steering angle

        // Constraint derivatives w.r.t. control [∂constraint/∂a, ∂constraint/∂delta]
        VecU acc_up_constraint_du;   acc_up_constraint_du <<  1.0,  0.0;
        VecU acc_low_constraint_du;  acc_low_constraint_du << -1.0,  0.0;
        VecU delta_up_constraint_du; delta_up_constraint_du <<  0.0,  1.0;
        VecU delta_low_constraint_du; delta_low_constraint_du <<  0.0, -1.0;

        // Constraint values
        double acc_up_constraint = get_bound_constr(a, A_MAX, true);
        double acc_low_constraint = get_bound_constr(a, A_MIN, false);
        double delta_up_constraint = get_bound_constr(delta, DELTA_MAX, true);
        double delta_low_constraint = get_bound_constr(delta, DELTA_MIN, false);

        // Sum of barrier Jacobians using chain rule
        // b_dot = q2 * b * c_dot where b = q1 * exp(q2 * c)
        lu = VecU::Zero();

        // Acceleration upper constraint
        double b = exp_barrier(acc_up_constraint);
        lu += Q2 * b * acc_up_constraint_du;

        // Acceleration lower constraint
        b = exp_barrier(acc_low_constraint);
        lu += Q2 * b * acc_low_constraint_du;

        // Steering upper constraint
        b = exp_barrier(delta_up_constraint);
        lu += Q2 * b * delta_up_constraint_du;

        // Steering lower constraint
        b = exp_barrier(delta_low_constraint);
        lu += Q2 * b * delta_low_constraint_du;

        return true;
    }

    /**
     * @brief Compute cost Hessian with respect to state
     *
     * Control constraints don't depend on state, so Hessian is zero.
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
        lxx = MatrixLXX::Zero();
        return true;
    }

    /**
     * @brief Compute cost Hessian with respect to control
     *
     * Applies chain rule through barrier functions:
     *   ∂²L/∂u² = Σ ∂²barrier/∂constraint² * ∂constraint/∂u * ∂constraint/∂u.T
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
        double a = ctrl(0);      // acceleration
        double delta = ctrl(1);  // steering angle

        // Constraint derivatives w.r.t. control
        VecU acc_up_constraint_du;   acc_up_constraint_du <<  1.0,  0.0;
        VecU acc_low_constraint_du;  acc_low_constraint_du << -1.0,  0.0;
        VecU delta_up_constraint_du; delta_up_constraint_du <<  0.0,  1.0;
        VecU delta_low_constraint_du; delta_low_constraint_du <<  0.0, -1.0;

        // Constraint values
        double acc_up_constraint = get_bound_constr(a, A_MAX, true);
        double acc_low_constraint = get_bound_constr(a, A_MIN, false);
        double delta_up_constraint = get_bound_constr(delta, DELTA_MAX, true);
        double delta_low_constraint = get_bound_constr(delta, DELTA_MIN, false);

        // Sum of barrier Hessians using chain rule
        // b_ddot = (q2^2) * b * (c_dot @ c_dot.T)
        luu = MatrixLUU::Zero();

        // Acceleration upper constraint
        double b = exp_barrier(acc_up_constraint);
        luu += (Q2 * Q2) * b * (acc_up_constraint_du * acc_up_constraint_du.transpose());

        // Acceleration lower constraint
        b = exp_barrier(acc_low_constraint);
        luu += (Q2 * Q2) * b * (acc_low_constraint_du * acc_low_constraint_du.transpose());

        // Steering upper constraint
        b = exp_barrier(delta_up_constraint);
        luu += (Q2 * Q2) * b * (delta_up_constraint_du * delta_up_constraint_du.transpose());

        // Steering lower constraint
        b = exp_barrier(delta_low_constraint);
        luu += (Q2 * Q2) * b * (delta_low_constraint_du * delta_low_constraint_du.transpose());

        return true;
    }

    /**
     * @brief Compute mixed cost Hessian (state x control)
     *
     * Control constraints don't depend on state, so mixed Hessian is zero.
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
     * @brief Get acceleration maximum bound
     * @return a_max in m/s²
     */
    static constexpr double get_a_max() { return A_MAX; }

    /**
     * @brief Get acceleration minimum bound
     * @return a_min in m/s²
     */
    static constexpr double get_a_min() { return A_MIN; }

    /**
     * @brief Get steering maximum bound
     * @return delta_max in radians
     */
    static constexpr double get_delta_max() { return DELTA_MAX; }

    /**
     * @brief Get steering minimum bound
     * @return delta_min in radians
     */
    static constexpr double get_delta_min() { return DELTA_MIN; }
};

} // namespace ilqr
