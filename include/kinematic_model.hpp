#pragma once

#include "model_base.hpp"
#include "config.hpp"

namespace ilqr {

template<typename T = double>
class KinematicModel : public Model<T, 4, 2> {
public:
    ILQR_PROBLEM_VARIABLES(T, 4, 2);

    KinematicModel()
        : wheelbase_(config::WHEELBASE),
          width_(config::WIDTH),
          length_(config::LENGTH) {}

    // Compute Jacobian of state transition w.r.t. state (A matrix)
    A gradient_fx(const State& state, const Control& ctrl, const T step = T(config::DT)) const override {
        T yaw = state(3);
        T v = state(2);
        T delta = ctrl(1);
        T beta_d = std::atan(std::tan(delta) / T(2));

        A dfdx;
        dfdx << T(1), T(0), std::cos(yaw + beta_d) * step, -v * std::sin(yaw + beta_d) * step,
                T(0), T(1), std::sin(yaw + beta_d) * step,  v * std::cos(yaw + beta_d) * step,
                T(0), T(0), T(1), T(0),
                T(0), T(0), T(2) * std::sin(beta_d) * step / wheelbase_, T(1);

        return dfdx;
    }

    // Compute Jacobian of state transition w.r.t. control (B matrix)
    B gradient_fu(const State& state, const Control& ctrl, const T step = T(config::DT)) const override {
        T delta = ctrl(1);
        T yaw = state(3);
        T v = state(2);
        T beta_d = std::atan(std::tan(delta) / T(2));
        T beta_over_stl = T(0.5) * (T(1) + std::tan(delta) * std::tan(delta)) /
                           (T(1) + T(0.25) * std::tan(delta) * std::tan(delta));

        B dfdu;
        dfdu << T(0), v * (-std::sin(beta_d + yaw)) * step * beta_over_stl,
                T(0), v * std::cos(beta_d + yaw) * step * beta_over_stl,
                step, T(0),
                T(0), (T(2) * v * step / wheelbase_) * std::cos(beta_d) * beta_over_stl;

        return dfdu;
    }

    // State transition function (kinematic bicycle model)
    State forward_calculation(const State& state, const Control& ctrl,
                             const T step = T(config::DT)) const override {
        T beta = std::atan(std::tan(ctrl(1)) / T(2));

        State next_x;
        next_x(0) = state(0) + state(2) * std::cos(beta + state(3)) * step;
        next_x(1) = state(1) + state(2) * std::sin(beta + state(3)) * step;
        next_x(2) = state(2) + ctrl(0) * step;
        next_x(3) = state(3) + T(2) * state(2) * std::sin(beta) * step / wheelbase_;

        return next_x;
    }

    // Get vehicle front and rear center positions
    std::pair<Position2D, Position2D> get_vehicle_front_and_rear_centers(
            const Position2D& pos, T yaw) const {
        T half_whba = T(0.5) * wheelbase_;
        Position2D half_whba_vec = half_whba * Position2D(std::cos(yaw), std::sin(yaw));

        Position2D front_pnt = pos + half_whba_vec;
        Position2D rear_pnt = pos - half_whba_vec;

        return {front_pnt, rear_pnt};
    }

    // Get derivatives of vehicle front and rear centers w.r.t. state
    std::pair<Matrix<T, 2, 4>, Matrix<T, 2, 4>> get_vehicle_front_and_rear_center_derivatives(T yaw) const {
        T half_whba = T(0.5) * wheelbase_;

        // Front point over state: [[x_fr->x, x_fr->y, x_fr->v, x_fr->yaw], [y_fr->x, y_fr->y, y_fr->v, y_fr->yaw]]
        Matrix<T, 2, 4> front_pnt_over_state;
        front_pnt_over_state << T(1), T(0), T(0), -half_whba * std::sin(yaw),
                               T(0), T(1), T(0),  half_whba * std::cos(yaw);

        // Rear point over state
        Matrix<T, 2, 4> rear_pnt_over_state;
        rear_pnt_over_state << T(1), T(0), T(0), half_whba * std::sin(yaw),
                              T(0), T(1), T(0), -half_whba * std::cos(yaw);

        return {front_pnt_over_state, rear_pnt_over_state};
    }

    // Initialize trajectory with zero controls
    States init_traj(const State& init_state, const Controls& controls, int horizon = config::HORIZON_LENGTH) {
        States states(horizon + 1);
        states[0] = init_state;

        for (int i = 1; i <= horizon; ++i) {
            states[i] = forward_calculation(states[i - 1], controls[i - 1]);
        }

        return states;
    }

    // Getters
    T get_wheelbase() const { return wheelbase_; }
    T get_width() const { return width_; }
    T get_length() const { return length_; }

private:
    T wheelbase_;
    T width_;
    T length_;
};

} // namespace ilqr
