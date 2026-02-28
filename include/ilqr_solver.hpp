/**
 * @file ilqr_solver.hpp
 * @brief iLQR (Iterative Linear Quadratic Regulator) solver
 *
 * Implements the iLQR algorithm for trajectory optimization with:
 * - Riccati backward recursion for computing gains
 * - Line search forward pass
 * - Adaptive regularization (lambda)
 */

#ifndef ILQR_ILQR_SOLVER_HPP
#define ILQR_ILQR_SOLVER_HPP

#include "common_variable.hpp"
#include "model_base.hpp"
#include "cost_calculator.hpp"
#include "kinematic_model.hpp"

#include <vector>
#include <memory>
#include <iostream>
#include <cmath>

namespace ilqr {

/**
 * @brief iLQR solver for trajectory optimization
 *
 * Iteratively linearizes the dynamics and quadraticizes the cost
 * to compute locally optimal feedback control policies.
 */
class ILQRSolver {
public:
    // Type definitions
    ILQR_PROBLEM_VARIABLES(double, 4, 2)

    // Dimension constants
    static constexpr int STATE_DIM = 4;
    static constexpr int CONTROL_DIM = 2;

    // Default parameters
    static constexpr int DEFAULT_HORIZON = 60;
    static constexpr double DEFAULT_DT = 0.1;
    static constexpr int DEFAULT_MAX_ITER = 50;
    static constexpr double DEFAULT_INIT_LAMBDA = 20.0;
    static constexpr double DEFAULT_LAMBDA_DECAY = 0.7;
    static constexpr double DEFAULT_LAMBDA_AMPLIFY = 2.0;
    static constexpr double DEFAULT_MAX_LAMBDA = 10000.0;
    static constexpr double DEFAULT_TOL = 0.001;

    /**
     * @brief Constructor
     *
     * @param model Kinematic model pointer
     * @param cost_calculator Cost function calculator
     * @param horizon Planning horizon
     */
    ILQRSolver(KinematicModel* model,
               CostCalculator* cost_calculator,
               int horizon = DEFAULT_HORIZON)
        : model_(model)
        , cost_calculator_(cost_calculator)
        , horizon_(horizon)
        , state_dim_(4)
        , control_dim_(2)
        , dt_(DEFAULT_DT)
        , max_iter_(DEFAULT_MAX_ITER)
        , init_lamb_(DEFAULT_INIT_LAMBDA)
        , lamb_decay_(DEFAULT_LAMBDA_DECAY)
        , lamb_amplify_(DEFAULT_LAMBDA_AMPLIFY)
        , max_lamb_(DEFAULT_MAX_LAMBDA)
        , tol_(DEFAULT_TOL)
    {
        // Initialize alpha options for line search
        alpha_options_ = {1.0, 0.5, 0.25, 0.125, 0.0625};
    }

    /**
     * @brief Backward pass: Riccati recursion to compute gains
     *
     * Computes feedforward gains (d) and feedback gains (K) by backward
     * recursion through time, using the Riccati equation.
     *
     * @param controls Current control trajectory
     * @param states Current state trajectory
     * @param lambda Regularization parameter
     * @param d Output: feedforward gains (horizon x 2)
     * @param K Output: feedback gains (horizon x 2x4)
     * @param delt_V Output: expected cost reduction
     * @return true if successful
     */
    bool backward_pass(const Controls& controls,
                       const States& states,
                       double lambda,
                       Controls& d,
                       MatrixCUXs& K,
                       double& delt_V) {
        // Get cost derivatives
        VecXs lx;
        MatrixCXXs lxx;
        VecUs lu;
        MatrixCUUs luu;
        MatrixCXUs lxu;

        cost_calculator_->CalculateDerivatives(states, controls, lx, lxx, lu, luu, lxu);

        // Initialize value function at terminal time
        VecX V_x = lx[horizon_];
        MatrixLXX V_xx = lxx[horizon_];

        // Resize outputs
        d.resize(horizon_);
        K.resize(horizon_);

        // Initialize expected cost reduction
        delt_V = 0.0;

        // Regularization matrix
        MatrixLXX regu_I = lambda * MatrixLXX::Identity();

        // Backward recursion
        for (int i = horizon_ - 1; i >= 0; --i) {
            // Get dynamics Jacobians
            MatrixLXX fx = model_->gradient_fx(states[i], controls[i]);
            MatrixLXU fu = model_->gradient_fu(states[i], controls[i]);

            // Q terms - using explicit computations to avoid Eigen template issues

            // Q_x = lx[i] + fx^T * V_x (4x1)
            VecX Q_x; Q_x.setZero();
            for (int r = 0; r < STATE_DIM; ++r) {
                Q_x(r) = lx[i](r);
                for (int k = 0; k < STATE_DIM; ++k) {
                    Q_x(r) += fx(k, r) * V_x(k);
                }
            }

            // Q_u = lu[i] + fu^T * V_x (2x1)
            VecU Q_u; Q_u.setZero();
            for (int r = 0; r < 2; ++r) {
                Q_u(r) = lu[i](r);
                for (int k = 0; k < STATE_DIM; ++k) {
                    Q_u(r) += fu(k, r) * V_x(k);
                }
            }

            // Q_xx = lxx[i] + fx^T * V_xx * fx (4x4)
            MatrixLXX Q_xx; Q_xx.setZero();
            // First compute temp = V_xx * fx (4x4 @ 4x4 = 4x4)
            MatrixLXX temp_Vxx_fx; temp_Vxx_fx.setZero();
            for (int r = 0; r < STATE_DIM; ++r) {
                for (int c = 0; c < 4; ++c) {
                    for (int k = 0; k < STATE_DIM; ++k) {
                        temp_Vxx_fx(r, c) += V_xx(r, k) * fx(k, c);
                    }
                }
            }
            // Then Q_xx = lxx + fx^T * temp (4x4)
            for (int r = 0; r < STATE_DIM; ++r) {
                for (int c = 0; c < 4; ++c) {
                    Q_xx(r, c) = lxx[i](r, c);
                    for (int k = 0; k < STATE_DIM; ++k) {
                        Q_xx(r, c) += fx(k, r) * temp_Vxx_fx(k, c);
                    }
                }
            }

            // Q_uu = luu[i] + fu^T * V_xx * fu (2x2)
            MatrixLUU Q_uu; Q_uu.setZero();
            // First compute temp = V_xx * fu (4x4 @ 4x2 = 4x2)
            MatrixLXU temp_Vxx_fu; temp_Vxx_fu.setZero();
            for (int r = 0; r < STATE_DIM; ++r) {
                for (int c = 0; c < CONTROL_DIM; ++c) {
                    for (int k = 0; k < STATE_DIM; ++k) {
                        temp_Vxx_fu(r, c) += V_xx(r, k) * fu(k, c);
                    }
                }
            }
            // Then Q_uu = luu + fu^T * temp (2x2)
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < CONTROL_DIM; ++c) {
                    Q_uu(r, c) = luu[i](r, c);
                    for (int k = 0; k < STATE_DIM; ++k) {
                        Q_uu(r, c) += fu(k, r) * temp_Vxx_fu(k, c);
                    }
                }
            }

            // Q_ux = lxu[i]^T + fu^T * V_xx * fx (2x4)
            // Note: lxu[i] is 4x2, so lxu[i]^T is 2x4
            MatrixLUX Q_ux; Q_ux.setZero();
            // Transpose lxu
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < 4; ++c) {
                    Q_ux(r, c) = lxu[i](c, r);
                }
            }
            // Add fu^T * temp_Vxx_fx (2x4)
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < 4; ++c) {
                    for (int k = 0; k < STATE_DIM; ++k) {
                        Q_ux(r, c) += fu(k, r) * temp_Vxx_fx(k, c);
                    }
                }
            }

            // Add regularization
            // Q_ux_regu = Q_ux + fu^T * regu_I * fx = Q_ux + (fu^T * regu_I) * fx
            // First compute fu^T * regu_I (2x4 @ 4x4 = 2x4)
            MatrixLUX fuT_regu; fuT_regu.setZero();
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < 4; ++c) {
                    for (int k = 0; k < STATE_DIM; ++k) {
                        fuT_regu(r, c) += fu(k, r) * regu_I(k, c);
                    }
                }
            }
            // Then Q_ux_regu = Q_ux + fuT_regu * fx (2x4 @ 4x4 = 2x4)
            MatrixLUX Q_ux_regu; Q_ux_regu.setZero();
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < 4; ++c) {
                    Q_ux_regu(r, c) = Q_ux(r, c);
                    for (int k = 0; k < STATE_DIM; ++k) {
                        Q_ux_regu(r, c) += fuT_regu(r, k) * fx(k, c);
                    }
                }
            }

            // Q_uu_regu = Q_uu + fu^T * regu_I * fu = Q_uu + fuT_regu * fu
            MatrixLUU Q_uu_regu; Q_uu_regu.setZero();
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < CONTROL_DIM; ++c) {
                    Q_uu_regu(r, c) = Q_uu(r, c);
                    for (int k = 0; k < STATE_DIM; ++k) {
                        Q_uu_regu(r, c) += fuT_regu(r, k) * fu(k, c);
                    }
                }
            }

            // Invert Q_uu_regu
            Eigen::ColPivHouseholderQR<MatrixLUU> qr(Q_uu_regu);
            if (!qr.isInvertible()) {
                return false;
            }
            MatrixLUU Q_uu_inv = qr.inverse();

            // Compute gains
            d[i] = -Q_uu_inv * Q_u;
            K[i] = -Q_uu_inv * Q_ux_regu;

            // Update value function (using explicit computations to avoid Eigen template issues)
            // V_x = Q_x + K.T @ Q_uu @ d + K.T @ Q_u + Q_ux.T @ d

            // Compute K.T @ Q_uu @ d
            VecX KtQ_uu_d; KtQ_uu_d.setZero();
            for (int r = 0; r < STATE_DIM; ++r) {
                for (int c = 0; c < CONTROL_DIM; ++c) {
                    KtQ_uu_d(r) += K[i](c, r) * (Q_uu(c, 0) * d[i](0) + Q_uu(c, 1) * d[i](1));
                }
            }

            // Compute K.T @ Q_u
            VecX KtQ_u; KtQ_u.setZero();
            for (int r = 0; r < STATE_DIM; ++r) {
                for (int c = 0; c < CONTROL_DIM; ++c) {
                    KtQ_u(r) += K[i](c, r) * Q_u(c);
                }
            }

            // Compute Q_ux.T @ d
            VecX Q_uxt_d; Q_uxt_d.setZero();
            for (int r = 0; r < STATE_DIM; ++r) {
                for (int c = 0; c < CONTROL_DIM; ++c) {
                    Q_uxt_d(r) += Q_ux(c, r) * d[i](c);
                }
            }

            V_x = Q_x + KtQ_uu_d + KtQ_u + Q_uxt_d;

            // V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K

            // Compute K.T @ Q_uu @ K (4x4 = 4x2 @ 2x2 @ 2x4)
            MatrixLXX KtQ_uu_K; KtQ_uu_K.setZero();
            for (int r = 0; r < STATE_DIM; ++r) {
                for (int c = 0; c < 4; ++c) {
                    for (int k = 0; k < CONTROL_DIM; ++k) {
                        for (int l = 0; l < 2; ++l) {
                            KtQ_uu_K(r, c) += K[i](k, r) * Q_uu(k, l) * K[i](l, c);
                        }
                    }
                }
            }

            // Compute K.T @ Q_ux (4x4 = 4x2 @ 2x4)
            MatrixLXX KtQ_ux; KtQ_ux.setZero();
            for (int r = 0; r < STATE_DIM; ++r) {
                for (int c = 0; c < 4; ++c) {
                    for (int k = 0; k < CONTROL_DIM; ++k) {
                        KtQ_ux(r, c) += K[i](k, r) * Q_ux(k, c);
                    }
                }
            }

            // Compute Q_ux.T @ K (4x4 = 4x2 @ 2x4)
            MatrixLXX Q_uxt_K; Q_uxt_K.setZero();
            for (int r = 0; r < STATE_DIM; ++r) {
                for (int c = 0; c < 4; ++c) {
                    for (int k = 0; k < CONTROL_DIM; ++k) {
                        Q_uxt_K(r, c) += Q_ux(k, r) * K[i](k, c);
                    }
                }
            }

            V_xx = Q_xx + KtQ_uu_K + KtQ_ux + Q_uxt_K;

            // Expected cost reduction (Python: 0.5 * d.T @ Q_uu @ d + d.T @ Q_u)
            double d_Q_uu_d = 0.0;
            for (int k = 0; k < CONTROL_DIM; ++k) {
                for (int l = 0; l < 2; ++l) {
                    d_Q_uu_d += d[i](k) * Q_uu(k, l) * d[i](l);
                }
            }
            double d_Q_u = d[i].transpose() * Q_u;
            delt_V += 0.5 * d_Q_uu_d + d_Q_u;
        }

        return true;
    }

    /**
     * @brief Forward pass: line search to apply gains
     *
     * Applies the computed gains with a given step size (alpha)
     * and forward propagates the dynamics.
     *
     * @param controls Current control trajectory
     * @param states Current state trajectory
     * @param d Feedforward gains
     * @param K Feedback gains
     * @param alpha Step size
     * @param new_controls Output: new control trajectory
     * @param new_states Output: new state trajectory
     * @return true if successful
     */
    bool forward_pass(const Controls& controls,
                      const States& states,
                      const Controls& d,
                      const MatrixCUXs& K,
                      double alpha,
                      Controls& new_controls,
                      States& new_states) {
        // Resize outputs
        new_controls.resize(horizon_);
        new_states.resize(horizon_ + 1);

        // Initial state
        new_states[0] = states[0];

        // Forward propagate
        for (int i = 0; i < horizon_; ++i) {
            // Compute new control
            State delta_x = new_states[i] - states[i];
            Control new_u = controls[i] + alpha * d[i] + K[i] * delta_x;
            new_controls[i] = new_u;

            // Forward dynamics
            new_states[i + 1] = model_->forward_calculation(new_states[i], new_controls[i]);
        }

        return true;
    }

    /**
     * @brief One iteration with line search and adaptive regularization
     *
     * Performs backward pass, then tries different alphas for line search.
     * Adjusts lambda based on whether cost reduction was achieved.
     *
     * @param controls Current control trajectory
     * @param states Current state trajectory
     * @param J Current cost
     * @param lambda Current regularization parameter
     * @param new_controls Output: new control trajectory
     * @param new_states Output: new state trajectory
     * @param new_J Output: new cost
     * @return true if iteration was effective (cost reduced)
     */
    bool iter(const Controls& controls,
              const States& states,
              double J,
              double& lambda,
              Controls& new_controls,
              States& new_states,
              double& new_J) {
        // Backward pass
        Controls d;
        MatrixCUXs K;
        double delt_V;

        if (!backward_pass(controls, states, lambda, d, K, delt_V)) {
            // Backward pass failed, amplify lambda
            lambda *= lamb_amplify_;
            return false;
        }

        // Try different alphas
        bool iter_effective = false;
        for (double alpha : alpha_options_) {
            if (forward_pass(controls, states, d, K, alpha, new_controls, new_states)) {
                new_J = cost_calculator_->CalculateTotalCost(new_states, new_controls);

                if (new_J < J) {
                    iter_effective = true;
                    break;
                }
            }
        }

        // Adjust lambda
        if (iter_effective) {
            lambda *= lamb_decay_;
        } else {
            lambda *= lamb_amplify_;
        }

        return iter_effective;
    }

    /**
     * @brief Main solver loop
     *
     * Iterates until convergence, max iterations, or lambda exceeds max.
     *
     * @param x0 Initial state
     * @param controls Output: optimal control trajectory
     * @param states Output: optimal state trajectory
     * @return true if converged successfully
     */
    bool solve(const State& x0, Controls& controls, States& states) {
        // Initialize with zero controls
        controls.resize(horizon_);
        Control zero_control = Control::Zero();
        for (int i = 0; i < horizon_; ++i) {
            controls[i] = zero_control;
        }

        // Forward propagate to get initial trajectory
        states = model_->init_traj(x0, controls);

        // Calculate initial cost
        double J = cost_calculator_->CalculateTotalCost(states, controls);

        // Initialize lambda
        double lambda = init_lamb_;

        std::cout << "Initial cost: " << J << ", Lambda: " << lambda << std::endl;

        // Main iteration loop
        for (int itr = 0; itr < max_iter_; ++itr) {
            Controls new_controls;
            States new_states;
            double new_J;
            bool iter_effective;

            iter_effective = iter(controls, states, J, lambda, new_controls, new_states, new_J);

            std::cout << "Iteration " << itr << ", Cost: " << new_J
                      << ", Lambda: " << lambda
                      << ", Effective: " << (iter_effective ? "Yes" : "No") << std::endl;

            if (iter_effective) {
                controls = new_controls;
                states = new_states;
                double J_old = J;
                J = new_J;

                // Check convergence
                if (std::abs(J - J_old) < tol_) {
                    std::cout << "Converged! Tolerance condition satisfied at iteration " << itr << std::endl;
                    return true;
                }
            } else {
                if (lambda > max_lamb_) {
                    std::cout << "Regularization parameter reached maximum." << std::endl;
                    return false;
                }
            }
        }

        std::cout << "Reached maximum iterations." << std::endl;
        return true;
    }

    // Getters
    int get_horizon() const { return horizon_; }
    double get_dt() const { return dt_; }
    int get_max_iter() const { return max_iter_; }
    double get_init_lamb() const { return init_lamb_; }
    double get_lamb_decay() const { return lamb_decay_; }
    double get_lamb_amplify() const { return lamb_amplify_; }
    double get_max_lamb() const { return max_lamb_; }
    double get_tol() const { return tol_; }

    // Setters
    void set_max_iter(int max_iter) { max_iter_ = max_iter; }
    void set_init_lamb(double init_lamb) { init_lamb_ = init_lamb; }
    void set_lamb_decay(double lamb_decay) { lamb_decay_ = lamb_decay; }
    void set_lamb_amplify(double lamb_amplify) { lamb_amplify_ = lamb_amplify; }
    void set_max_lamb(double max_lamb) { max_lamb_ = max_lamb; }
    void set_tol(double tol) { tol_ = tol; }
    void set_alpha_options(const std::vector<double>& alphas) { alpha_options_ = alphas; }

private:
    KinematicModel* model_;
    CostCalculator* cost_calculator_;

    int horizon_;
    int state_dim_;
    int control_dim_;
    double dt_;

    // Solver parameters
    int max_iter_;
    double init_lamb_;
    double lamb_decay_;
    double lamb_amplify_;
    double max_lamb_;
    double tol_;
    std::vector<double> alpha_options_;
};

} // namespace ilqr

#endif // ILQR_ILQR_SOLVER_HPP
