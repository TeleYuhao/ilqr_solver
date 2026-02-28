/**
 * @file test_obstacle_verification.cpp
 * @brief Verification test for obstacle.hpp vs Python obstacle_base.py
 *
 * This test compares the C++ implementation with Python reference values
 * to ensure numerical consistency (tolerance: 1e-8).
 */

#include "../include/obstacle.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>

using namespace ilqr;

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
bool compare_vector_2d(const std::string& name, const Eigen::Vector2d& cpp_vec,
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
 * @brief Compare two 4D states with tolerance
 */
bool compare_state_4d(const std::string& name, const Eigen::Matrix<double, 4, 1>& cpp_state,
                      const double py_ref[4]) {
    std::cout << name << ":" << std::endl;
    bool all_pass = true;

    for (int i = 0; i < 4; ++i) {
        std::stringstream ss;
        ss << "    [" << i << "]";
        all_pass &= compare_double(ss.str(), cpp_state(i), py_ref[i]);
    }

    return all_pass;
}

// ============================================================================
// Test 1: Constructor and ellipsoid scales
// ============================================================================
void test_constructor_and_scales() {
    std::cout << "\n=== Test 1: Constructor and ellipsoid scales ===" << std::endl;

    // Create obstacle
    Eigen::Matrix<double, 4, 1> state;
    state << 5.0, -0.2, 3.0, -0.0;  // [x, y, v, yaw]

    Eigen::Matrix<double, 3, 1> attr;
    attr << 2.0, 4.5, 1.5;  // [width, length, safety_buffer] - stored as column vector

    Obstacle obs(state, attr);

    // Python reference values
    double py_a = 4.750000000000000;
    double py_b = 3.500000000000000;

    compare_double("Ellipsoid scale a", obs.get_a(), py_a);
    compare_double("Ellipsoid scale b", obs.get_b(), py_b);
}

// ============================================================================
// Test 2: ellipsoid_safety_margin()
// ============================================================================
void test_ellipsoid_safety_margin() {
    std::cout << "\n=== Test 2: ellipsoid_safety_margin() ===" << std::endl;

    // Create obstacle
    Eigen::Matrix<double, 4, 1> state;
    state << 5.0, -0.2, 3.0, -0.0;

    Eigen::Matrix<double, 3, 1> attr;
    attr << 2.0, 4.5, 1.5;

    Obstacle obs(state, attr);

    // Test inputs
    Eigen::Vector2d pnt;
    pnt << 5.5, 0.0;

    Eigen::Vector2d elp_center;
    elp_center << 5.0, -0.2;

    double margin = obs.ellipsoid_safety_margin(pnt, elp_center);

    // Python reference value
    double py_ref = 0.985654361467579;

    compare_double("Safety margin", margin, py_ref);
}

// ============================================================================
// Test 3: ellipsoid_safety_margin_derivatives()
// ============================================================================
void test_ellipsoid_safety_margin_derivatives() {
    std::cout << "\n=== Test 3: ellipsoid_safety_margin_derivatives() ===" << std::endl;

    // Create obstacle
    Eigen::Matrix<double, 4, 1> state;
    state << 5.0, -0.2, 3.0, -0.0;

    Eigen::Matrix<double, 3, 1> attr;
    attr << 2.0, 4.5, 1.5;

    Obstacle obs(state, attr);

    // Test inputs
    Eigen::Vector2d pnt;
    pnt << 5.5, 0.0;

    Eigen::Vector2d elp_center;
    elp_center << 5.0, -0.2;

    auto deriv = obs.ellipsoid_safety_margin_derivatives(pnt, elp_center);

    // Python reference values
    double py_ref[2] = {-0.044321329639889, -0.032653061224490};

    compare_vector_2d("Derivatives", deriv, py_ref);
}

// ============================================================================
// Test 4: const_velo_prediction()
// ============================================================================
void test_const_velo_prediction() {
    std::cout << "\n=== Test 4: const_velo_prediction() ===" << std::endl;

    // Initial state
    Eigen::Matrix<double, 4, 1> x0;
    x0 << 5.0, -0.2, 3.0, -0.0;

    int steps = 5;

    // Call static method
    auto pred_traj = Obstacle::const_velo_prediction(x0, steps);

    // Check a few key states
    std::cout << "Checking first 3 states:" << std::endl;

    double py_state0[4] = {5.0, -0.2, 3.0, -0.0};
    double py_state1[4] = {5.3, -0.2, 3.0, -0.0};
    double py_state5[4] = {6.5, -0.2, 3.0, -0.0};

    compare_state_4d("  state[0]", pred_traj[0], py_state0);
    compare_state_4d("  state[1]", pred_traj[1], py_state1);
    compare_state_4d("  state[5]", pred_traj[5], py_state5);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << std::fixed;
    std::cout << std::setprecision(15);

    std::cout << "============================================" << std::endl;
    std::cout << "Obstacle C++ vs Python Verification" << std::endl;
    std::cout << "============================================" << std::endl;

    // Parse command line arguments
    int test_to_run = -1;  // -1 means run all tests
    if (argc > 1) {
        test_to_run = std::atoi(argv[1]);
    }

    if (test_to_run == -1 || test_to_run == 1) test_constructor_and_scales();
    if (test_to_run == -1 || test_to_run == 2) test_ellipsoid_safety_margin();
    if (test_to_run == -1 || test_to_run == 3) test_ellipsoid_safety_margin_derivatives();
    if (test_to_run == -1 || test_to_run == 4) test_const_velo_prediction();

    // Print summary
    std::cout << "\n============================================" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  Passed: " << GREEN << tests_passed << RESET << std::endl;
    std::cout << "  Failed: " << (tests_failed > 0 ? RED : "") << tests_failed << RESET << std::endl;
    std::cout << "============================================" << std::endl;

    return (tests_failed == 0) ? 0 : 1;
}
