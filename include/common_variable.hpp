#pragma once

#include <Eigen/Core>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

// 定义 ILQR 问题相关类型的宏
#define ILQR_PROBLEM_VARIABLES(T, M, N) \
    typedef Matrix<T, M, 1> State; \
    typedef Matrix<T, N, 1> Control; \
    typedef Matrix<T, 2, 1> Position2D; \
    typedef Matrix<T, M, 1> VecX; \
    typedef Matrix<T, N, 1> VecU; \
    typedef Matrix<T, M, M> A; \
    typedef Matrix<T, M, N> B; \
    typedef Matrix<T, M, M> MatrixLXX; \
    typedef Matrix<T, N, N> MatrixLUU; \
    typedef Matrix<T, M, N> MatrixLXU; \
    typedef Matrix<T, N, M> MatrixLUX; \
    typedef Matrix<T, M + N, N + N> MatrixCF; \
    typedef std::vector<State, Eigen::aligned_allocator<State>> States; \
    typedef std::vector<Control, Eigen::aligned_allocator<Control>> Controls; \
    typedef std::vector<VecX, Eigen::aligned_allocator<VecX>> VecXs; \
    typedef std::vector<VecU, Eigen::aligned_allocator<VecU>> VecUs; \
    typedef std::vector<MatrixLXX, Eigen::aligned_allocator<MatrixLXX>> MatrixCXXs; \
    typedef std::vector<MatrixLUU, Eigen::aligned_allocator<MatrixLUU>> MatrixCUUs; \
    typedef std::vector<MatrixLXU, Eigen::aligned_allocator<MatrixLXU>> MatrixCXUs; \
    typedef std::vector<MatrixLUX, Eigen::aligned_allocator<MatrixLUX>> MatrixCUXs;
