#pragma once

#include "common_variable.hpp"
#include <Eigen/Core>
#include <cmath>

namespace ilqr {

/**
 * @brief Abstract base class for iLQR cost functions
 *
 * Template class defining the interface for cost functions used in iLQR.
 * Corresponds to Python's CostFunc class in cost_base.py.
 *
 * @tparam T Scalar type (e.g., double, float)
 * @tparam M State dimension
 * @tparam N Control dimension
 */
template <typename T, int M, int N>
class CostFunc {
public:
    // Type definitions from the macro
    ILQR_PROBLEM_VARIABLES(T, M, N)

    // Constants for dimensions
    static constexpr int STATE_DIM = M;
    static constexpr int CONTROL_DIM = N;

    // Default parameters for exponential barrier functions
    static constexpr T DEFAULT_Q1 = T(5.5);
    static constexpr T DEFAULT_Q2 = T(5.75);

    /**
     * @brief Constructor
     * @param state_dim State dimension (M)
     * @param control_dim Control dimension (N)
     */
    explicit CostFunc(int state_dim = M, int control_dim = N)
        : M_(state_dim), N_(control_dim), horizon_(0) {}

    /**
     * @brief Virtual destructor for proper inheritance
     */
    virtual ~CostFunc() = default;

    /**
     * @brief Compute cost function value
     *
     * @param step Time step
     * @param state Current state, shape (M,)
     * @param ctrl Current control, shape (N,)
     * @param val Output parameter for cost value
     * @return true if computation successful
     */
    virtual inline bool value(int step,
                              const State& state,
                              const Control& ctrl,
                              T& val) const = 0;

    /**
     * @brief Compute cost function gradient with respect to state
     *
     * @param step Time step
     * @param state Current state, shape (M,)
     * @param ctrl Current control, shape (N,)
     * @param lx Output parameter for gradient, shape (M,)
     * @return true if computation successful
     */
    virtual inline bool gradient_lx(int step,
                                    const State& state,
                                    const Control& ctrl,
                                    VecX& lx) const = 0;

    /**
     * @brief Compute cost function gradient with respect to control
     *
     * @param step Time step
     * @param state Current state, shape (M,)
     * @param ctrl Current control, shape (N,)
     * @param lu Output parameter for gradient, shape (N,)
     * @return true if computation successful
     */
    virtual inline bool gradient_lu(int step,
                                    const State& state,
                                    const Control& ctrl,
                                    VecU& lu) const = 0;

    /**
     * @brief Compute cost function Hessian with respect to state
     *
     * @param step Time step
     * @param state Current state, shape (M,)
     * @param ctrl Current control, shape (N,)
     * @param lxx Output parameter for Hessian, shape (M, M)
     * @return true if computation successful
     */
    virtual inline bool hessian_lxx(int step,
                                    const State& state,
                                    const Control& ctrl,
                                    MatrixLXX& lxx) const = 0;

    /**
     * @brief Compute cost function Hessian with respect to control
     *
     * @param step Time step
     * @param state Current state, shape (M,)
     * @param ctrl Current control, shape (N,)
     * @param luu Output parameter for Hessian, shape (N, N)
     * @return true if computation successful
     */
    virtual inline bool hessian_luu(int step,
                                    const State& state,
                                    const Control& ctrl,
                                    MatrixLUU& luu) const = 0;

    /**
     * @brief Compute cost function mixed Hessian (state x control)
     *
     * @param step Time step
     * @param state Current state, shape (M,)
     * @param ctrl Current control, shape (N,)
     * @param lxu Output parameter for Hessian, shape (M, N)
     * @return true if computation successful
     */
    virtual inline bool hessian_lxu(int step,
                                    const State& state,
                                    const Control& ctrl,
                                    MatrixLXU& lxu) const = 0;

    /**
     * @brief Set prediction horizon length
     * @param step Horizon length
     */
    inline void set_horizon(int step) { horizon_ = step; }

    /**
     * @brief Get prediction horizon length
     * @return Horizon length
     */
    inline int get_horizon() const { return horizon_; }

    /**
     * @brief Get state dimension
     * @return State dimension M
     */
    inline int state_dim() const { return M_; }

    /**
     * @brief Get control dimension
     * @return Control dimension N
     */
    inline int control_dim() const { return N_; }

    /**
     * @brief Exponential barrier function
     *
     * Computes b = q1 * exp(q2 * c)
     *
     * @param c Constraint value
     * @param q1 First barrier parameter (default 5.5)
     * @param q2 Second barrier parameter (default 5.75)
     * @return Barrier value
     */
    static inline T exp_barrier(T c, T q1 = DEFAULT_Q1, T q2 = DEFAULT_Q2) {
        return q1 * std::exp(q2 * c);
    }

    /**
     * @brief Exponential barrier Jacobian (chain rule)
     *
     * Computes db/dc = q2 * b * c_dot
     *
     * @param c Constraint value
     * @param c_dot Constraint gradient
     * @param q1 First barrier parameter (default 5.5)
     * @param q2 Second barrier parameter (default 5.75)
     * @return Barrier derivative
     */
    static inline VecX exp_barrier_jacobian(const T& c,
                                             const VecX& c_dot,
                                             T q1 = DEFAULT_Q1,
                                             T q2 = DEFAULT_Q2) {
        T b = exp_barrier(c, q1, q2);
        return q2 * b * c_dot;
    }

    /**
     * @brief Exponential barrier Hessian (chain rule)
     *
     * Computes d²b/dc² = (q2²) * b * (c_dot @ c_dot.T)
     *
     * @param c Constraint value
     * @param c_dot Constraint gradient
     * @param q1 First barrier parameter (default 5.5)
     * @param q2 Second barrier parameter (default 5.75)
     * @return Barrier Hessian
     */
    static inline MatrixLXX exp_barrier_hessian(const T& c,
                                                 const VecX& c_dot,
                                                 T q1 = DEFAULT_Q1,
                                                 T q2 = DEFAULT_Q2) {
        T b = exp_barrier(c, q1, q2);
        return (q2 * q2) * b * (c_dot * c_dot.transpose());
    }

    /**
     * @brief Get bound constraint value
     *
     * Computes constraint for upper or lower bound.
     * - Upper bound: var - bound (should be <= 0)
     * - Lower bound: bound - var (should be <= 0)
     *
     * @param var Variable value
     * @param bound Bound value
     * @param is_upper_bound True for upper bound, false for lower bound
     * @return Constraint value
     */
    static inline T get_bound_constr(T var, T bound, bool is_upper_bound = true) {
        if (is_upper_bound) {
            return var - bound;
        } else {
            return bound - var;
        }
    }

protected:
    int M_;          ///< State dimension
    int N_;          ///< Control dimension
    int horizon_;    ///< Prediction horizon length
};

} // namespace ilqr
