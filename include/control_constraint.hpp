#pragma once

#include "cost_base.hpp"
#include "config.hpp"

namespace ilqr {

template<typename T = double>
class ControlConstraint : public CostFunc<T, 4, 2> {
public:
    ILQR_PROBLEM_VARIABLES(T, 4, 2);

    ControlConstraint() = default;

    bool value(int step, const State& state, const Control& ctrl, T& val) const override {
        T a = ctrl(0);
        T delta = ctrl(1);

        // Acceleration constraints
        T acc_up_constr = this->exp_barrier(
            this->get_bound_constr(a, config::A_MAX, true));
        T acc_low_constr = this->exp_barrier(
            this->get_bound_constr(a, config::A_MIN, false));

        // Steering angle constraints
        T delta_up_constr = this->exp_barrier(
            this->get_bound_constr(delta, config::DELTA_MAX, true));
        T delta_low_constr = this->exp_barrier(
            this->get_bound_constr(delta, config::DELTA_MIN, false));

        val = acc_up_constr + acc_low_constr + delta_up_constr + delta_low_constr;
        return true;
    }

    bool gradient_lx(int step, const State& state, const Control& ctrl, VecX& lx) const override {
        lx = VecX::Zero();
        return true;
    }

    bool gradient_lu(int step, const State& state, const Control& ctrl, VecU& lu) const override {
        T a = ctrl(0);
        T delta = ctrl(1);

        // Acceleration constraint derivatives (da/da = 1, da/ddelta = 0)
        T acc_up_constr = this->get_bound_constr(a, config::A_MAX, true);
        T acc_low_constr = this->get_bound_constr(a, config::A_MIN, false);
        T acc_up_grad = config::EXP_Q2 * this->exp_barrier(acc_up_constr);
        T acc_low_grad = -config::EXP_Q2 * this->exp_barrier(acc_low_constr);  // Negative for lower bound

        // Steering constraint derivatives (ddelta/da = 0, ddelta/ddelta = 1)
        T delta_up_constr = this->get_bound_constr(delta, config::DELTA_MAX, true);
        T delta_low_constr = this->get_bound_constr(delta, config::DELTA_MIN, false);
        T delta_up_grad = config::EXP_Q2 * this->exp_barrier(delta_up_constr);
        T delta_low_grad = -config::EXP_Q2 * this->exp_barrier(delta_low_constr);  // Negative for lower bound

        lu << acc_up_grad + acc_low_grad, delta_up_grad + delta_low_grad;

        return true;
    }

    bool hessian_lxx(int step, const State& state, const Control& ctrl, MatrixLXX& lxx) const override {
        lxx = MatrixLXX::Zero();
        return true;
    }

    bool hessian_luu(int step, const State& state, const Control& ctrl, MatrixLUU& luu) const override {
        T a = ctrl(0);
        T delta = ctrl(1);

        // Acceleration constraint second derivatives
        VecU acc_up_constr_du;
        acc_up_constr_du << T(1), T(0);
        VecU acc_low_constr_du;
        acc_low_constr_du << T(-1), T(0);

        T acc_up_constr = this->get_bound_constr(a, config::A_MAX, true);
        T acc_low_constr = this->get_bound_constr(a, config::A_MIN, false);
        T acc_up_hess = config::EXP_Q2 * config::EXP_Q2 * this->exp_barrier(acc_up_constr);
        T acc_low_hess = config::EXP_Q2 * config::EXP_Q2 * this->exp_barrier(acc_low_constr);

        // Steering constraint second derivatives
        VecU delta_up_constr_du;
        delta_up_constr_du << T(0), T(1);
        VecU delta_low_constr_du;
        delta_low_constr_du << T(0), T(-1);

        T delta_up_constr = this->get_bound_constr(delta, config::DELTA_MAX, true);
        T delta_low_constr = this->get_bound_constr(delta, config::DELTA_MIN, false);
        T delta_up_hess = config::EXP_Q2 * config::EXP_Q2 * this->exp_barrier(delta_up_constr);
        T delta_low_hess = config::EXP_Q2 * config::EXP_Q2 * this->exp_barrier(delta_low_constr);

        luu = acc_up_hess * (acc_up_constr_du * acc_up_constr_du.transpose()) +
              acc_low_hess * (acc_low_constr_du * acc_low_constr_du.transpose()) +
              delta_up_hess * (delta_up_constr_du * delta_up_constr_du.transpose()) +
              delta_low_hess * (delta_low_constr_du * delta_low_constr_du.transpose());

        return true;
    }

    bool hessian_lxu(int step, const State& state, const Control& ctrl, MatrixLXU& lxu) const override {
        lxu = MatrixLXU::Zero();
        return true;
    }
};

} // namespace ilqr
