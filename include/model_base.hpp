#pragma once

#include "common_variable.hpp"

template<typename T, int M, int N>
class Model {
public:
    ILQR_PROBLEM_VARIABLES(T, M, N);  // 展开所有类型定义
    Model() = default;
    virtual ~Model() = default;

    virtual inline A gradient_fx(const State& state,
                                 const Control& ctrl,
                                 const T step) const = 0;

    virtual inline B gradient_fu(const State& state,
                                 const Control& ctrl,
                                 const T step) const = 0;

    virtual inline State forward_calculation(const State& state,
                                             const Control& ctrl,
                                             const T step) const = 0;

    inline void set_timer(const T timer) {
        timer_ = timer;
    }

    inline void update_timer(const T dt) {
        timer_ += dt;
    }

    inline T get_timer() const {
        return timer_;
    }

protected:
    T timer_{T(0)};
};
