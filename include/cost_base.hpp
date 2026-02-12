#pragma once

#include "common_variable.hpp"

template<typename T, int M, int N>
class CostFunc {
public:
    ILQR_PROBLEM_VARIABLES(T, M, N);
    CostFunc() = default;
    virtual ~CostFunc() = default;

    void set_horizon(int step) {
        horizon = step;
    }

    int get_horizon() const {
        return horizon;
    }

    virtual bool value(int step,
                       const State& state,
                       const Control& ctrl,
                       double& val) const = 0;

    /* gradient matrix
     * [df_1/dx_1, df_1/dx_2, ... df_1/dx_j, ...]
     * [df_2/dx_1, df_2/dx_2, ... df_2/dx_j, ...]
     * [   :        ...       ...             : ]
     * [df_i/dx_1, df_i/dx_2, ... df_i/dx_j, ...]
     * g_ij = df_i/dx_j
     */
    virtual bool gradient_lx(int step,
                             const State& state,
                             const Control& ctrl,
                             VecX& lx) const = 0;

    virtual bool gradient_lu(int step,
                             const State& state,
                             const Control& ctrl,
                             VecU& lu) const = 0;

    /* H_ij = d(df_i/dx_i)/d_x_j
     */
    virtual bool hessian_lxx(int step,
                             const State& state,
                             const Control& ctrl,
                             MatrixLXX& lxx) const = 0;

    virtual bool hessian_luu(int step,
                             const State& state,
                             const Control& ctrl,
                             MatrixLUU& luu) const = 0;

    virtual bool hessian_lxu(int step,
                             const State& state,
                             const Control& ctrl,
                             MatrixLXU& lxu) const = 0;

    // Exponential barrier functions for constraint handling
    static inline double exp_barrier(double c, double q1 = 5.5, double q2 = 5.75) {
        return q1 * std::exp(q2 * c);
    }

    static inline VecX exp_barrier_jacobian(double c, const VecX& c_dot,
                                            double q1 = 5.5, double q2 = 5.75) {
        double b = exp_barrier(c, q1, q2);
        return q2 * b * c_dot;
    }

    static inline MatrixLXX exp_barrier_hessian(double c, const VecX& c_dot,
                                               double q1 = 5.5, double q2 = 5.75) {
        double b = exp_barrier(c, q1, q2);
        return (q2 * q2 * b) * (c_dot * c_dot.transpose());
    }

    static inline double get_bound_constr(double var, double bound, bool upper) {
        return upper ? (var - bound) : (bound - var);
    }

protected:
    int horizon{0};
};
