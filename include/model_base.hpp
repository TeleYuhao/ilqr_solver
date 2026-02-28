#pragma once

#include "common_variable.hpp"
#include <Eigen/Core>
#include <chrono>

namespace ilqr {

/**
 * @brief Abstract base class for iLQR models
 *
 * Template class defining the interface for dynamical models used in iLQR.
 * Corresponds to Python's Model class in model_base.py.
 *
 * @tparam T Scalar type (e.g., double, float)
 * @tparam M State dimension
 * @tparam N Control dimension
 */
template <typename T, int M, int N>
class Model {
public:
    // Type definitions from the macro
    ILQR_PROBLEM_VARIABLES(T, M, N)

    // Constants for dimensions
    static constexpr int STATE_DIM = M;
    static constexpr int CONTROL_DIM = N;

    /**
     * @brief Constructor
     * @param state_dim State dimension (M)
     * @param control_dim Control dimension (N)
     */
    explicit Model(int state_dim = M, int control_dim = N)
        : M_(state_dim), N_(control_dim), timer_(T(0)) {}

    /**
     * @brief Virtual destructor for proper inheritance
     */
    virtual ~Model() = default;

    /**
     * @brief Compute state transition Jacobian with respect to state (A matrix)
     *
     * Computes df/dx where f is the dynamics function.
     *
     * @param state Current state, shape (M,)
     * @param ctrl Current control, shape (N,)
     * @param step Time step
     * @return A matrix, shape (M, M)
     */
    virtual inline A gradient_fx(const State& state,
                                  const Control& ctrl,
                                  const T step) const = 0;

    /**
     * @brief Compute state transition Jacobian with respect to control (B matrix)
     *
     * Computes df/du where f is the dynamics function.
     *
     * @param state Current state, shape (M,)
     * @param ctrl Current control, shape (N,)
     * @param step Time step
     * @return B matrix, shape (M, N)
     */
    virtual inline B gradient_fu(const State& state,
                                  const Control& ctrl,
                                  const T step) const = 0;

    /**
     * @brief Forward calculation (system dynamics)
     *
     * Computes the next state given current state and control.
     *
     * @param state Current state, shape (M,)
     * @param ctrl Current control, shape (N,)
     * @param step Time step
     * @return Next state, shape (M,)
     */
    virtual inline State forward_calculation(const State& state,
                                             const Control& ctrl,
                                             const T step) const = 0;

    /**
     * @brief Set timer value
     * @param timer Timer value to set
     */
    inline void set_timer(const T timer) { timer_ = timer; }

    /**
     * @brief Update timer by adding dt
     * @param dt Time increment
     */
    inline void update_timer(const T dt) { timer_ += dt; }

    /**
     * @brief Get current timer value
     * @return Current timer value
     */
    inline T get_timer() const { return timer_; }

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

protected:
    int M_;          ///< State dimension
    int N_;          ///< Control dimension
    T timer_;        ///< Timer value
};

/**
 * @brief Timer utility class for performance measurement
 */
class Timer {
public:
    /**
     * @brief Constructor, starts the timer
     */
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    /**
     * @brief Reset the timer
     */
    inline void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Get elapsed time in milliseconds
     * @return Elapsed time in milliseconds
     */
    inline double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return duration.count() / 1000.0;
    }

    /**
     * @brief Get elapsed time in seconds
     * @return Elapsed time in seconds
     */
    inline double elapsed_s() const {
        return elapsed_ms() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace ilqr
