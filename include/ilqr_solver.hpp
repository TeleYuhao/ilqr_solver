#pragma once

#include "model_base.hpp"
#include "cost_calculator.hpp"
#include "config.hpp"
#include <iostream>
#include <cmath>

namespace ilqr {

template<typename T = double>
class ILQRSolver {
public:
    typedef Model<T, 4, 2> KinematicModelType;
    typedef CostCalculator<T> CostCalculatorType;

    // Matrix types for trajectory storage (row-major: N+1 x 4 for states, N x 2 for controls)
    typedef Matrix<T, Dynamic, 4> States;
    typedef Matrix<T, Dynamic, 2> Controls;
    typedef Matrix<T, 4, 1> VecX;
    typedef Matrix<T, 2, 1> VecU;
    typedef Matrix<T, 4, 4> MatrixLXX;
    typedef Matrix<T, 2, 2> MatrixLUU;
    typedef Matrix<T, 4, 2> MatrixLXU;

    // Vector types for cost derivatives (aligned allocator for Eigen types in std::vector)
    typedef std::vector<VecX, Eigen::aligned_allocator<VecX>> VecXs;
    typedef std::vector<VecU, Eigen::aligned_allocator<VecU>> VecUs;
    typedef std::vector<MatrixLXX, Eigen::aligned_allocator<MatrixLXX>> MatrixCXXs;
    typedef std::vector<MatrixLUU, Eigen::aligned_allocator<MatrixLUU>> MatrixCUUs;
    typedef std::vector<MatrixLXU, Eigen::aligned_allocator<MatrixLXU>> MatrixCXUs;

    ILQRSolver(KinematicModelType* model, CostCalculatorType* cost_func)
        : model_(model), cost_func_(cost_func),
          N_(config::HORIZON_LENGTH), dt_(config::DT),
          max_iter_(config::MAX_ITER),
          init_lamb_(config::INIT_LAMB),
          lamb_decay_(config::LAMB_DECAY),
          lamb_amplify_(config::LAMB_AMPLIFY),
          max_lamb_(config::MAX_LAMB),
          tol_(config::TOL) {

        alpha_options_ = {T(1), T(0.5), T(0.25), T(0.125), T(0.0625)};
    }

    // Test access methods (for comprehensive testing)
    std::tuple<Controls, std::vector<Matrix<T, 2, 4>>, T>
    test_backward_pass(const Controls& u, const States& x, T lamb, int debug_iter = -1) {
        return backward_pass(u, x, lamb, debug_iter);
    }

    std::pair<Controls, States>
    test_forward_pass(const Controls& u, const States& x,
                     const std::vector<Matrix<T, 2, 4>>& K,
                     const Controls& d, T alpha) {
        return forward_pass(u, x, d, K, alpha);
    }

    std::tuple<Controls, States, T, bool>
    test_iter(const Controls& u, const States& x, T J, T lamb, int current_iter = 0) {
        return iter(u, x, J, lamb, current_iter);
    }

    std::pair<Controls, States> solve(const typename KinematicModelType::State& x0) {
        // Initialize with zero controls
        Controls init_u = Controls::Zero(N_, 2);
        States init_x = States::Zero(N_ + 1, 4);
        init_x.row(0) = x0.transpose();

        // Propagate dynamics forward to get initial trajectory
        for (int i = 0; i < N_; ++i) {
            typename KinematicModelType::State state_i = init_x.row(i).transpose();
            typename KinematicModelType::Control ctrl_i = init_u.row(i).transpose();
            typename KinematicModelType::State next_state = model_->forward_calculation(state_i, ctrl_i, dt_);
            init_x.row(i + 1) = next_state.transpose();
        }

        // Convert to vector types for cost calculation (use aligned_allocator)
        std::vector<typename KinematicModelType::State, Eigen::aligned_allocator<typename KinematicModelType::State>> state_vec(N_ + 1);
        std::vector<typename KinematicModelType::Control, Eigen::aligned_allocator<typename KinematicModelType::Control>> ctrl_vec(N_);
        for (int i = 0; i <= N_; ++i) {
            state_vec[i] = init_x.row(i).transpose();
        }
        for (int i = 0; i < N_; ++i) {
            ctrl_vec[i] = init_u.row(i).transpose();
        }

        T J = cost_func_->CalculateTotalCost(state_vec, ctrl_vec);
        Controls u = init_u;
        States x = init_x;

        T lamb = init_lamb_;

        for (int itr = 0; itr < max_iter_; ++itr) {
            auto [new_u, new_x, new_J, effective] = iter(u, x, J, lamb, itr);

            std::cout << "Iteration " << itr << ", Cost: " << new_J
                      << ", Lambda: " << lamb << ", Effective: " << effective << std::endl;

            if (effective) {
                x = new_x;
                u = new_u;
                T J_old = J;
                J = new_J;

                if (std::abs(J - J_old) < tol_) {
                    std::cout << "Tolerance condition satisfied. " << itr << std::endl;
                    break;
                }

                lamb *= lamb_decay_;
            } else {
                lamb *= lamb_amplify_;

                if (lamb > max_lamb_) {
                    std::cout << "Regularization parameter reached maximum." << std::endl;
                    break;
                }
            }
        }

        return {u, x};
    }

private:
    std::tuple<Controls, States, T, bool>
    iter(const Controls& u, const States& x, T J, T lamb, int current_iter = 0) {
        auto [d, K, exp_redu] = backward_pass(u, x, lamb, current_iter);

        bool iter_effective_flag = false;
        Controls new_u = Controls::Zero(N_, 2);
        States new_x = States::Zero(N_ + 1, 4);
        T new_J = std::numeric_limits<T>::max();

        for (T alpha : alpha_options_) {
            auto [u_updated, x_updated] = forward_pass(u, x, d, K, alpha);

            // Convert to vector types for cost calculation (use aligned_allocator)
            std::vector<typename KinematicModelType::State, Eigen::aligned_allocator<typename KinematicModelType::State>> state_vec(N_ + 1);
            std::vector<typename KinematicModelType::Control, Eigen::aligned_allocator<typename KinematicModelType::Control>> ctrl_vec(N_);
            for (int i = 0; i <= N_; ++i) {
                state_vec[i] = x_updated.row(i).transpose();
            }
            for (int i = 0; i < N_; ++i) {
                ctrl_vec[i] = u_updated.row(i).transpose();
            }

            new_J = cost_func_->CalculateTotalCost(state_vec, ctrl_vec);

            if (new_J < J) {
                new_u = u_updated;
                new_x = x_updated;
                iter_effective_flag = true;
                break;
            }
        }

        return {new_u, new_x, new_J, iter_effective_flag};
    }

    std::tuple<Controls, std::vector<Matrix<T, 2, 4>>, T>
    backward_pass(const Controls& u, const States& x, T lamb, int debug_iter = -1) {
        // Update reference states and calculate derivatives
        Matrix<T, Eigen::Dynamic, 2> positions(x.rows(), 2);
        for (int i = 0; i < x.rows(); ++i) {
            positions.row(i) << x(i, 0), x(i, 1);
        }
        cost_func_->getStateCost()->get_ref_states(positions);

        // Convert matrix types to vector types for cost calculation (use aligned_allocator)
        std::vector<typename KinematicModelType::State, Eigen::aligned_allocator<typename KinematicModelType::State>> state_vec(N_ + 1);
        std::vector<typename KinematicModelType::Control, Eigen::aligned_allocator<typename KinematicModelType::Control>> ctrl_vec(N_);
        for (int i = 0; i <= N_; ++i) {
            state_vec[i] = x.row(i).transpose();
        }
        for (int i = 0; i < N_; ++i) {
            ctrl_vec[i] = u.row(i).transpose();
        }

        VecXs lx;
        MatrixCXXs lxx;
        VecUs lu;
        MatrixCUUs luu;
        MatrixCXUs lxu;
        cost_func_->CalculateDerivates(state_vec, ctrl_vec, lx, lxx, lu, luu, lxu);

        // Initialize value function from terminal cost
        VecX V_x = lx[N_];  // Terminal state gradient
        MatrixLXX V_xx = lxx[N_];  // Terminal state Hessian

        // Feedback and feedforward gains
        Controls d = Controls::Zero(N_, 2);
        std::vector<Matrix<T, 2, 4>> K(N_);
        for (int i = 0; i < N_; ++i) {
            K[i] = Matrix<T, 2, 4>::Zero();
        }

        T delt_V = T(0);
        MatrixLXX regu_I = lamb * MatrixLXX::Identity();

        // Backward pass from N-1 to 0
        for (int i = N_ - 1; i >= 0; --i) {
            typename KinematicModelType::State state_i = x.row(i).transpose();
            typename KinematicModelType::Control ctrl_i = u.row(i).transpose();

            Matrix<T, 4, 4> dfdx = model_->gradient_fx(state_i, ctrl_i, dt_);
            Matrix<T, 4, 2> dfdu = model_->gradient_fu(state_i, ctrl_i, dt_);

            // Q-function terms
            VecX Q_x = lx[i] + dfdx.transpose() * V_x;
            VecU Q_u = lu[i] + dfdu.transpose() * V_x;
            MatrixLXX Q_xx = lxx[i] + dfdx.transpose() * V_xx * dfdx;
            MatrixLUU Q_uu = luu[i] + dfdu.transpose() * V_xx * dfdu;
            Matrix<T, 2, 4> Q_ux = lxu[i].transpose() + dfdu.transpose() * V_xx * dfdx;

            // Compute gains with regularization
            Matrix<T, 2, 4> dfdu_regu = dfdu.transpose() * regu_I;
            Matrix<T, 2, 4> Q_ux_regu = Q_ux + dfdu_regu * dfdx;
            MatrixLUU Q_uu_regu = Q_uu + dfdu_regu * dfdu;

            // Use LDLT decomposition for better numerical stability (matches Python's np.linalg.inv)
            Eigen::LDLT<MatrixLUU> ldlt(Q_uu_regu);
            MatrixLUU Q_uu_inv = MatrixLUU::Identity();
            ldlt.solveInPlace(Q_uu_inv);

            d.row(i) = -Q_uu_inv * Q_u;
            K[i] = -Q_uu_inv * Q_ux_regu;

            // Update value function
            V_x = Q_x + K[i].transpose() * Q_uu * d.row(i).transpose() +
                  K[i].transpose() * Q_u + Q_ux.transpose() * d.row(i).transpose();
            V_xx = Q_xx + K[i].transpose() * Q_uu * K[i] +
                   K[i].transpose() * Q_ux + Q_ux.transpose() * K[i];

            // Expected cost reduction (extract scalar from 1x1 matrix)
            T term1 = (T(0.5) * d.row(i) * Q_uu * d.row(i).transpose()).value();
            T term2 = (d.row(i) * Q_u).value();
            delt_V += term1 + term2;

            // Debug output for last step of backward pass (step 59)
            if (debug_iter == 0 && i == N_ - 1) {
                std::cout << "=== Backward Pass Debug (iter=" << debug_iter << ", step=" << i << ") ===" << std::endl;
                std::cout << "Q_uu_regu:\n" << Q_uu_regu << std::endl;
                std::cout << "Q_uu_inv:\n" << Q_uu_inv << std::endl;
                std::cout << "d.row(" << i << "):\n" << d.row(i) << std::endl;
                std::cout << "K[" << i << "]:\n" << K[i] << std::endl;
                std::cout << "Q_x:\n" << Q_x.transpose() << std::endl;
                std::cout << "Q_u:\n" << Q_u.transpose() << std::endl;
                std::cout << "V_x (after update):\n" << V_x.transpose() << std::endl;
                std::cout << "V_xx (after update):\n" << V_xx << std::endl;
            }
        }

        return {d, std::move(K), delt_V};
    }

    std::pair<Controls, States> forward_pass(const Controls& u, const States& x,
                                             const Controls& d,
                                             const std::vector<Matrix<T, 2, 4>>& K,
                                             T alpha) {
        Controls new_u = Controls::Zero(N_, 2);
        States new_x = States::Zero(N_ + 1, 4);
        new_x.row(0) = x.row(0);

        for (int i = 0; i < N_; ++i) {
            VecU new_u_i = u.row(i).transpose() +
                             alpha * d.row(i).transpose() +
                             K[i] * (new_x.row(i).transpose() - x.row(i).transpose());
            new_u.row(i) = new_u_i.transpose();

            // Convert row to State type for forward_calculation
            typename KinematicModelType::State state_i = new_x.row(i).transpose();
            typename KinematicModelType::Control ctrl_i = new_u.row(i).transpose();
            typename KinematicModelType::State next_state = model_->forward_calculation(state_i, ctrl_i, dt_);
            new_x.row(i + 1) = next_state.transpose();
        }

        return {new_u, new_x};
    }

    KinematicModelType* model_;
    CostCalculatorType* cost_func_;

    // Parameters
    int N_;
    T dt_;
    int max_iter_;
    T init_lamb_;
    T lamb_decay_;
    T lamb_amplify_;
    T max_lamb_;
    std::vector<T> alpha_options_;
    T tol_;
};

} // namespace ilqr
