#pragma once

#include "common_variable.hpp"
#include "config.hpp"
#include <vector>

namespace ilqr {

template<typename T = double>
class Obstacle {
public:
    typedef Matrix<T, 4, 1> State;
    typedef Matrix<T, 2, 1> Position2D;

    Obstacle(const State& state, const Matrix<T, 3, 1>& attr)
        : state_(state), attr_(attr), yaw_(state(3)) {

        ego_width_ = config::EGO_WIDTH;
        ego_pnt_radius_ = config::EGO_PNT_RADIUS;

        obs_width_ = attr_(0);
        obs_length_ = attr_(1);
        d_safe_ = attr_(2);

        // Compute ellipsoid semi-axes
        a_ = T(0.5) * obs_length_ + d_safe_ + ego_pnt_radius_;
        b_ = T(0.5) * obs_width_ + d_safe_ + ego_pnt_radius_;

        // Generate prediction trajectory
        prediction_traj_ = const_velo_prediction(state_, config::HORIZON_LENGTH);
    }

    // Compute ellipsoid safety margin (positive = safe, negative = collision)
    T ellipsoid_safety_margin(const Position2D& pnt, const Position2D& elp_center) const {
        T theta = yaw_;

        Position2D diff = pnt - elp_center;
        Matrix<T, 2, 2> rotation_matrix;
        rotation_matrix << std::cos(theta), -std::sin(theta),
                           std::sin(theta),  std::cos(theta);

        Position2D pnt_std = rotation_matrix.transpose() * diff;

        return T(1) - ((pnt_std(0) * pnt_std(0)) / (a_ * a_) +
                        (pnt_std(1) * pnt_std(1)) / (b_ * b_));
    }

    // Compute gradient of safety margin w.r.t. point
    Position2D ellipsoid_safety_margin_derivatives(const Position2D& pnt,
                                                   const Position2D& elp_center) const {
        T theta = yaw_;

        Position2D diff = pnt - elp_center;
        Matrix<T, 2, 2> rotation_matrix;
        rotation_matrix << std::cos(theta), -std::sin(theta),
                           std::sin(theta),  std::cos(theta);

        Position2D pnt_std = rotation_matrix.transpose() * diff;

        // (1) constraint over standard point vector
        Position2D res_over_pnt_std;
        res_over_pnt_std(0) = T(-2) * pnt_std(0) / (a_ * a_);
        res_over_pnt_std(1) = T(-2) * pnt_std(1) / (b_ * b_);

        // (2) standard point vector over difference vector = rotation_matrix^T
        // (3) difference vector over original point = I

        // Chain rule: (1) @ (2) @ (3)
        // Python (row vectors): res_over_pnt_std @ rotation_matrix.T @ I
        // Result is a (1x2) row vector = [c/dx, c/dy]
        // C++ (column vectors): same as (rotation_matrix @ res_over_pnt_std)
        // This gives a (2x1) column vector, which is the transpose of the Python result
        // But the function returns Position2D (column vector), so we need to TRANSPOSE the result
        Position2D res_over_pnt_col = rotation_matrix * res_over_pnt_std;
        Position2D res_over_pnt = res_over_pnt_col;  // Column vector format for C++

        return res_over_pnt;
    }

    // Generate constant velocity prediction trajectory
    std::vector<State> const_velo_prediction(const State& x0, int steps) const {
        std::vector<State> predicted_states;
        predicted_states.reserve(steps + 1);
        predicted_states.push_back(x0);

        State cur_x = x0;
        for (int i = 0; i < steps; ++i) {
            cur_x = kinematic_propagate(cur_x);
            predicted_states.push_back(cur_x);
        }

        return predicted_states;
    }

    // Member access
    const std::vector<State>& get_prediction_traj() const { return prediction_traj_; }
    T get_a() const { return a_; }
    T get_b() const { return b_; }
    const State& get_state() const { return state_; }
    T get_yaw() const { return yaw_; }

private:
    // Kinematic propagation for prediction
    State kinematic_propagate(const State& x0) const {
        T x = x0(0), y = x0(1), v = x0(2), yaw = x0(3);
        T dt = config::DT;

        State next_x;
        next_x(0) = x + v * std::cos(yaw) * dt;
        next_x(1) = y + v * std::sin(yaw) * dt;
        next_x(2) = v;
        next_x(3) = yaw;

        return next_x;
    }

    State state_;
    Matrix<T, 3, 1> attr_;
    T ego_width_;
    T ego_pnt_radius_;
    T obs_width_, obs_length_, d_safe_;
    T a_, b_;  // Ellipsoid semi-axes
    T yaw_;
    std::vector<State> prediction_traj_;
};

} // namespace ilqr
