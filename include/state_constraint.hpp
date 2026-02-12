#pragma once

#include "cost_base.hpp"
#include "config.hpp"
#include "obstacle.hpp"
#include "kinematic_model.hpp"
#include <vector>

namespace ilqr {

template<typename T = double>
class StateConstraint : public CostFunc<T, 4, 2> {
public:
    ILQR_PROBLEM_VARIABLES(T, 4, 2);

    StateConstraint(const KinematicModel<T>* model,
                    const std::vector<Obstacle<T>>& obstacles)
        : model_(model), obstacles_(obstacles) {}

    bool value(int step, const State& state, const Control& ctrl, T& val) const override {
        T x = state(0), y = state(1), v = state(2), yaw = state(3);

        // Velocity constraints
        T velo_up_constr = this->exp_barrier(
            this->get_bound_constr(v, config::V_MAX, true));
        T velo_down_constr = this->exp_barrier(
            this->get_bound_constr(v, config::V_MIN, false));

        val = velo_down_constr + velo_up_constr;

        // Obstacle avoidance
        Position2D pos;
        pos << x, y;
        auto [front_pnt, rear_pnt] = model_->get_vehicle_front_and_rear_centers(pos, yaw);

        for (const auto& obs : obstacles_) {
            const auto& pred_traj = obs.get_prediction_traj();
            if (step >= 0 && step < pred_traj.size()) {
                Position2D obs_center;
                obs_center << pred_traj[step](0), pred_traj[step](1);

                val += this->exp_barrier(obs.ellipsoid_safety_margin(front_pnt, obs_center));
                val += this->exp_barrier(obs.ellipsoid_safety_margin(rear_pnt, obs_center));
            }
        }

        return true;
    }

    bool gradient_lx(int step, const State& state, const Control& ctrl, VecX& lx) const override {
        T x = state(0), y = state(1), v = state(2), yaw = state(3);

        lx = VecX::Zero();

        // Velocity constraint gradients
        T velo_up_constr = this->get_bound_constr(v, config::V_MAX, true);
        T velo_low_constr = this->get_bound_constr(v, config::V_MIN, false);

        VecX velo_up_constr_dx;
        velo_up_constr_dx << T(0), T(0), T(1), T(0);
        VecX velo_low_constr_dx;
        velo_low_constr_dx << T(0), T(0), T(-1), T(0);

        lx += this->exp_barrier_jacobian(velo_up_constr, velo_up_constr_dx);
        lx += this->exp_barrier_jacobian(velo_low_constr, velo_low_constr_dx);

        // Obstacle avoidance gradients
        Position2D pos;
        pos << x, y;
        auto [front_pnt, rear_pnt] = model_->get_vehicle_front_and_rear_centers(pos, yaw);

        for (const auto& obs : obstacles_) {
            const auto& pred_traj = obs.get_prediction_traj();
            if (step >= 0 && step < pred_traj.size()) {
                Position2D obs_center;
                obs_center << pred_traj[step](0), pred_traj[step](1);

                T front = obs.ellipsoid_safety_margin(front_pnt, obs_center);
                T rear = obs.ellipsoid_safety_margin(rear_pnt, obs_center);

                Position2D front_safety_over_ego_front =
                    obs.ellipsoid_safety_margin_derivatives(front_pnt, obs_center);
                Position2D rear_safety_over_ego_rear =
                    obs.ellipsoid_safety_margin_derivatives(rear_pnt, obs_center);

                auto [ego_front_over_state, ego_rear_over_state] =
                    model_->get_vehicle_front_and_rear_center_derivatives(yaw);

                // Chain rule: d_safety/d_state = d_safety/d_point * d_point/d_state
                // Full matrix multiplication: (1x2) * (2x4) = (1x4)
                VecX front_safety_over_state = front_safety_over_ego_front.transpose() * ego_front_over_state;
                front_safety_over_state(2) = T(0);  // No dependency on v

                VecX rear_safety_over_state = rear_safety_over_ego_rear.transpose() * ego_rear_over_state;
                rear_safety_over_state(2) = T(0);  // No dependency on v

                lx += this->exp_barrier_jacobian(front, front_safety_over_state);
                lx += this->exp_barrier_jacobian(rear, rear_safety_over_state);
            }
        }

        return true;
    }

    bool gradient_lu(int step, const State& state, const Control& ctrl, VecU& lu) const override {
        lu = VecU::Zero();
        return true;
    }

    bool hessian_lxx(int step, const State& state, const Control& ctrl, MatrixLXX& lxx) const override {
        T x = state(0), y = state(1), v = state(2), yaw = state(3);

        lxx = MatrixLXX::Zero();

        // Velocity constraint Hessians
        T velo_up_constr = this->get_bound_constr(v, config::V_MAX, true);
        T velo_low_constr = this->get_bound_constr(v, config::V_MIN, false);

        VecX velo_up_constr_dx;
        velo_up_constr_dx << T(0), T(0), T(1), T(0);
        VecX velo_low_constr_dx;
        velo_low_constr_dx << T(0), T(0), T(-1), T(0);

        lxx += this->exp_barrier_hessian(velo_up_constr, velo_up_constr_dx);
        lxx += this->exp_barrier_hessian(velo_low_constr, velo_low_constr_dx);

        // Obstacle avoidance Hessians
        Position2D pos;
        pos << x, y;
        auto [front_pnt, rear_pnt] = model_->get_vehicle_front_and_rear_centers(pos, yaw);

        for (const auto& obs : obstacles_) {
            const auto& pred_traj = obs.get_prediction_traj();
            if (step >= 0 && step < pred_traj.size()) {
                Position2D obs_center;
                obs_center << pred_traj[step](0), pred_traj[step](1);

                T front = obs.ellipsoid_safety_margin(front_pnt, obs_center);
                T rear = obs.ellipsoid_safety_margin(rear_pnt, obs_center);

                Position2D front_safety_over_ego_front =
                    obs.ellipsoid_safety_margin_derivatives(front_pnt, obs_center);
                Position2D rear_safety_over_ego_rear =
                    obs.ellipsoid_safety_margin_derivatives(rear_pnt, obs_center);

                auto [ego_front_over_state, ego_rear_over_state] =
                    model_->get_vehicle_front_and_rear_center_derivatives(yaw);

                // Chain rule: d_safety/d_state = d_safety/d_point * d_point/d_state
                // Full matrix multiplication: (1x2) * (2x4) = (1x4)
                VecX front_safety_over_state = front_safety_over_ego_front.transpose() * ego_front_over_state;
                front_safety_over_state(2) = T(0);  // No dependency on v

                VecX rear_safety_over_state = rear_safety_over_ego_rear.transpose() * ego_rear_over_state;
                rear_safety_over_state(2) = T(0);  // No dependency on v

                lxx += this->exp_barrier_hessian(front, front_safety_over_state);
                lxx += this->exp_barrier_hessian(rear, rear_safety_over_state);
            }
        }

        return true;
    }

    bool hessian_luu(int step, const State& state, const Control& ctrl, MatrixLUU& luu) const override {
        luu = MatrixLUU::Zero();
        return true;
    }

    bool hessian_lxu(int step, const State& state, const Control& ctrl, MatrixLXU& lxu) const override {
        lxu = MatrixLXU::Zero();
        return true;
    }

private:
    const KinematicModel<T>* model_;
    std::vector<Obstacle<T>> obstacles_;
};

} // namespace ilqr
