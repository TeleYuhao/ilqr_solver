#pragma once

#include "cost_base.hpp"
#include "config.hpp"

namespace ilqr {

template<typename T = double>
class StateCost : public CostFunc<T, 4, 2> {
public:
    ILQR_PROBLEM_VARIABLES(T, 4, 2);

    typedef Matrix<T, 2, Eigen::Dynamic> RefWaypoints;

    StateCost(const MatrixLXX& Q, const MatrixLUU& R, const RefWaypoints& ref_waypoints)
        : Q_(Q), R_(R), ref_waypoints_(ref_waypoints),
          ref_velo_(config::REF_VELO), horizon_(config::HORIZON_LENGTH) {
        ref_states_.resize(horizon_ + 1);
    }

    // Match positions to reference waypoints (find nearest)
    void get_ref_states(const Matrix<T, Eigen::Dynamic, 2>& pos) {
        int n_waypoints = ref_waypoints_.cols();
        int n_steps = pos.rows();

        // Resize ref_states_ to match the number of steps
        if (static_cast<int>(ref_states_.size()) != n_steps) {
            ref_states_.resize(n_steps);
        }

        // Reshape ref_waypoints for broadcasting: (2, n_waypoints, 1)
        std::vector<Matrix<T, 2, 1>> ref_reshaped(n_waypoints);
        for (int i = 0; i < n_waypoints; ++i) {
            ref_reshaped[i] = ref_waypoints_.col(i);
        }

        // Find nearest reference point for each position
        for (int i = 0; i < n_steps; ++i) {
            Position2D p = pos.row(i).transpose();
            int min_idx = 0;
            T min_dist = (ref_reshaped[0] - p).squaredNorm();

            for (int j = 1; j < n_waypoints; ++j) {
                T dist = (ref_reshaped[j] - p).squaredNorm();
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = j;
                }
            }

            ref_states_[i](0) = ref_reshaped[min_idx](0);
            ref_states_[i](1) = ref_reshaped[min_idx](1);
            ref_states_[i](2) = ref_velo_;
            ref_states_[i](3) = T(0);
        }
    }

    bool value(int step, const State& state, const Control& ctrl, T& val) const override {
        if (step < 0 || step >= ref_states_.size()) {
            return false;
        }

        State ref_state = ref_states_[step];
        State state_diff = state - ref_state;
        // Extract scalar from 1x1 matrix expression using .value()
        val = (state_diff.transpose() * Q_ * state_diff).value() +
              (ctrl.transpose() * R_ * ctrl).value();
        return true;
    }

    bool gradient_lx(int step, const State& state, const Control& ctrl, VecX& lx) const override {
        if (step < 0 || step >= ref_states_.size()) {
            return false;
        }

        State ref_state = ref_states_[step];
        State state_diff = state - ref_state;
        lx = T(2) * (Q_ * state_diff);
        return true;
    }

    bool gradient_lu(int step, const State& state, const Control& ctrl, VecU& lu) const override {
        lu = T(2) * (R_ * ctrl);
        return true;
    }

    bool hessian_lxx(int step, const State& state, const Control& ctrl, MatrixLXX& lxx) const override {
        lxx = T(2) * Q_;
        return true;
    }

    bool hessian_luu(int step, const State& state, const Control& ctrl, MatrixLUU& luu) const override {
        luu = T(2) * R_;
        return true;
    }

    bool hessian_lxu(int step, const State& state, const Control& ctrl, MatrixLXU& lxu) const override {
        lxu = MatrixLXU::Zero();
        return true;
    }

    // Getters
    const States& get_ref_states() const { return ref_states_; }
    T get_ref_velo() const { return ref_velo_; }

private:
    MatrixLXX Q_;
    MatrixLUU R_;
    RefWaypoints ref_waypoints_;
    States ref_states_;
    T ref_velo_;
    int horizon_;
};

} // namespace ilqr
