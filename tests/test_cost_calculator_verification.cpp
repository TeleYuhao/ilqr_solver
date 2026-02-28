/**
 * @file test_cost_calculator_verification.cpp
 * @brief Verification test for cost_calculator.hpp vs Python CostCalculator.py
 *
 * This test compares the C++ implementation with Python reference values
 * to ensure numerical consistency (tolerance: 1e-8).
 */

#include "../include/cost_calculator.hpp"
#include "../include/kinematic_model.hpp"
#include "../include/obstacle.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <memory>

using namespace ilqr;

// Type definitions (matching ILQR_PROBLEM_VARIABLES macro)
ILQR_PROBLEM_VARIABLES(double, 4, 2)

// Tolerance for numerical comparison
constexpr double TOLERANCE = 1e-8;

// Color codes for terminal output
const char* GREEN = "\033[32m";
const char* RED = "\033[31m";
const char* RESET = "\033[0m";

// Test result tracking
int tests_passed = 0;
int tests_failed = 0;

/**
 * @brief Compare two doubles with tolerance
 */
bool compare_double(const std::string& name, double cpp_val, double py_ref, int precision = 15) {
    double diff = std::abs(cpp_val - py_ref);
    bool pass = diff < TOLERANCE;

    std::cout << "  " << name << ": ";
    std::cout << std::setprecision(precision) << cpp_val << " (C++) vs " << py_ref << " (Python)";
    std::cout << " [Diff: " << diff << "]";

    if (pass) {
        std::cout << " " << GREEN << "PASS" << RESET << std::endl;
        ++tests_passed;
    } else {
        std::cout << " " << RED << "FAIL" << RESET << std::endl;
        ++tests_failed;
    }

    return pass;
}

/**
 * @brief Compare two 2D vectors with tolerance
 */
bool compare_vector_2d(const std::string& name, const Eigen::Matrix<double, 2, 1>& cpp_vec,
                       const double py_ref[2]) {
    std::cout << name << ":" << std::endl;
    bool all_pass = true;

    for (int i = 0; i < 2; ++i) {
        std::stringstream ss;
        ss << "    [" << i << "]";
        all_pass &= compare_double(ss.str(), cpp_vec(i), py_ref[i]);
    }

    return all_pass;
}

/**
 * @brief Compare two 4D vectors with tolerance
 */
bool compare_vector_4d(const std::string& name, const Eigen::Matrix<double, 4, 1>& cpp_vec,
                       const double py_ref[4]) {
    std::cout << name << ":" << std::endl;
    bool all_pass = true;

    for (int i = 0; i < 4; ++i) {
        std::stringstream ss;
        ss << "    [" << i << "]";
        all_pass &= compare_double(ss.str(), cpp_vec(i), py_ref[i]);
    }

    return all_pass;
}

/**
 * @brief Compare 4x4 matrix with tolerance
 */
bool compare_matrix_4x4(const std::string& name, const Eigen::Matrix4d& cpp_mat,
                        const double py_ref[16]) {
    std::cout << name << ":" << std::endl;
    bool all_pass = true;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::stringstream ss;
            ss << "    [" << i << "," << j << "]";
            all_pass &= compare_double(ss.str(), cpp_mat(i, j), py_ref[i * 4 + j]);
        }
    }

    return all_pass;
}

/**
 * @brief Compare 2x2 matrix with tolerance
 */
bool compare_matrix_2x2(const std::string& name, const Eigen::Matrix2d& cpp_mat,
                        const double py_ref[4]) {
    std::cout << name << ":" << std::endl;
    bool all_pass = true;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::stringstream ss;
            ss << "    [" << i << "," << j << "]";
            all_pass &= compare_double(ss.str(), cpp_mat(i, j), py_ref[i * 2 + j]);
        }
    }

    return all_pass;
}

// ============================================================================
// Test 1: CalculateTotalCost()
// ============================================================================
void test_calculate_total_cost() {
    std::cout << "\n=== Test 1: CalculateTotalCost() ===" << std::endl;

    // Create model
    auto model = std::make_shared<KinematicModel>();

    // Create obstacles
    Eigen::Matrix<double, 4, 1> obs_state;
    obs_state << 15.0, 2.0, 0.0, 0.0;

    Eigen::Matrix<double, 3, 1> obs_attr;
    obs_attr << 2.0, 4.5, 1.5;

    auto obs = std::make_shared<Obstacle>(obs_state, obs_attr);
    std::vector<std::shared_ptr<Obstacle>> obstacles;
    obstacles.push_back(obs);

    // Create reference waypoints (2 x 1000) - matching Python format
    int num_waypoints = 1000;
    Eigen::Matrix<double, 2, Eigen::Dynamic> ref_waypoints(2, num_waypoints);
    for (int i = 0; i < num_waypoints; ++i) {
        double x = 50.0 * i / (num_waypoints - 1);  // 0 to 50
        double y = 0.0;
        ref_waypoints(0, i) = x;
        ref_waypoints(1, i) = y;
    }

    // Create CostCalculator
    CostCalculator cost_calculator(model.get(), obstacles, ref_waypoints, 60);

    // Create test trajectory
    State x0;
    x0 << 0.0, 0.0, 5.0, 0.0;

    int horizon = 60;
    Controls init_controls;
    init_controls.reserve(horizon);
    Control zero_control = Control::Zero();
    for (int i = 0; i < horizon; ++i) {
        init_controls.push_back(zero_control);
    }
    States states = model->init_traj(x0, init_controls);

    // Test CalculateTotalCost
    double total_cost = cost_calculator.CalculateTotalCost(states, init_controls);

    // Python reference value (matching test.py Q = diag([1.0, 1.0, 0.5, 0])): 3742.879969678473117
    double py_ref = 3742.879969678473117;

    compare_double("Total cost", total_cost, py_ref);
}

// ============================================================================
// Test 2: CalculateDerivatives() - Sample values
// ============================================================================
void test_calculate_derivatives() {
    std::cout << "\n=== Test 2: CalculateDerivatives() ===" << std::endl;

    // Create model
    auto model = std::make_shared<KinematicModel>();

    // Create obstacles
    Eigen::Matrix<double, 4, 1> obs_state;
    obs_state << 15.0, 2.0, 0.0, 0.0;

    Eigen::Matrix<double, 3, 1> obs_attr;
    obs_attr << 2.0, 4.5, 1.5;

    auto obs = std::make_shared<Obstacle>(obs_state, obs_attr);
    std::vector<std::shared_ptr<Obstacle>> obstacles;
    obstacles.push_back(obs);

    // Create reference waypoints (2 x 1000)
    int num_waypoints = 1000;
    Eigen::Matrix<double, 2, Eigen::Dynamic> ref_waypoints(2, num_waypoints);
    for (int i = 0; i < num_waypoints; ++i) {
        double x = 50.0 * i / (num_waypoints - 1);
        double y = 0.0;
        ref_waypoints(0, i) = x;
        ref_waypoints(1, i) = y;
    }

    // Create CostCalculator
    CostCalculator cost_calculator(model.get(), obstacles, ref_waypoints, 60);

    // Create test trajectory
    State x0;
    x0 << 0.0, 0.0, 5.0, 0.0;

    int horizon = 60;
    Controls init_controls;
    init_controls.reserve(horizon);
    Control zero_control = Control::Zero();
    for (int i = 0; i < horizon; ++i) {
        init_controls.push_back(zero_control);
    }
    States states = model->init_traj(x0, init_controls);

    // Test CalculateDerivatives
    VecXs lx;
    MatrixCXXs lxx;
    VecUs lu;
    MatrixCUUs luu;
    MatrixCXUs lxu;

    cost_calculator.CalculateDerivatives(states, init_controls, lx, lxx, lu, luu, lxu);

    // Check sizes
    std::cout << "  Output sizes:" << std::endl;
    std::cout << "    lx size: " << lx.size() << " (expected 61)" << std::endl;
    std::cout << "    lxx size: " << lxx.size() << " (expected 61)" << std::endl;
    std::cout << "    lu size: " << lu.size() << " (expected 60)" << std::endl;
    std::cout << "    luu size: " << luu.size() << " (expected 60)" << std::endl;
    std::cout << "    lxu size: " << lxu.size() << " (expected 60)" << std::endl;

    if (lx.size() == 61 && lxx.size() == 61 && lu.size() == 60 &&
        luu.size() == 60 && lxu.size() == 60) {
        std::cout << "    " << GREEN << "PASS" << RESET << std::endl;
        ++tests_passed;
    } else {
        std::cout << "    " << RED << "FAIL" << RESET << std::endl;
        ++tests_failed;
    }

    // Check lx at t=0
    double py_lx_0[4] = {0.0, 0.0, -1.0, 0.0};
    compare_vector_4d("lx at t=0", lx[0], py_lx_0);

    // Check lx at t=1
    double py_lx_1[4] = {-0.001001001000999, 0.000000000000001, -1.0, 0.000000000000001};
    compare_vector_4d("lx at t=1", lx[1], py_lx_1);

    // Check lx at t=20 (near obstacle)
    double py_lx_20[4] = {31.701205079116193, 36.511566604409943, -1.0, 65.707197707884660};
    compare_vector_4d("lx at t=20", lx[20], py_lx_20);

    // Check lu at t=0 (should be zeros)
    double py_lu_0[2] = {0.0, 0.0};
    compare_vector_2d("lu at t=0", lu[0], py_lu_0);

    // Check luu at t=0
    double py_luu_0[4] = {
        2.003684188415652, 0.0,
        0.0, 2.043665148266215
    };
    compare_matrix_2x2("luu at t=0", luu[0], py_luu_0);

    // Check luu at t=30 (should be constant)
    double py_luu_30[4] = {
        2.003684188415652, 0.0,
        0.0, 2.043665148266215
    };
    compare_matrix_2x2("luu at t=30", luu[30], py_luu_30);

    // Check lxx diagonal at t=0
    std::cout << "  lxx at t=0 diagonal:" << std::endl;
    double py_lxx_0_diag[4] = {2.0, 2.0, 1.000000000118785, 0.0};
    for (int i = 0; i < 4; ++i) {
        std::stringstream ss;
        ss << "    [" << i << "," << i << "]";
        compare_double(ss.str(), lxx[0](i, i), py_lxx_0_diag[i]);
    }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << std::fixed;
    std::cout << std::setprecision(15);

    std::cout << "============================================" << std::endl;
    std::cout << "CostCalculator C++ vs Python Verification" << std::endl;
    std::cout << "============================================" << std::endl;

    // Parse command line arguments
    int test_to_run = -1;
    if (argc > 1) {
        test_to_run = std::atoi(argv[1]);
    }

    if (test_to_run == -1 || test_to_run == 1) test_calculate_total_cost();
    if (test_to_run == -1 || test_to_run == 2) test_calculate_derivatives();

    // Print summary
    std::cout << "\n============================================" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  Passed: " << GREEN << tests_passed << RESET << std::endl;
    std::cout << "  Failed: " << (tests_failed > 0 ? RED : "") << tests_failed << RESET << std::endl;
    std::cout << "============================================" << std::endl;

    return (tests_failed == 0) ? 0 : 1;
}
