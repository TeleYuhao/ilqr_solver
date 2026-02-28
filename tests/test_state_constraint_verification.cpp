/**
 * @file test_state_constraint_verification.cpp
 * @brief Verification test for state_constraint.hpp vs Python StateConstraint.py
 *
 * This test compares the C++ implementation with Python reference values
 * to ensure numerical consistency (tolerance: 1e-8).
 */

#include "../include/state_constraint.hpp"
#include "../include/kinematic_model.hpp"
#include "../include/obstacle.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <memory>

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

/**
 * @brief Compare 4x2 matrix with tolerance
 */
bool compare_matrix_4x2(const std::string& name, const Eigen::Matrix<double, 4, 2>& cpp_mat,
                        const double py_ref[8]) {
    std::cout << name << ":" << std::endl;
    bool all_pass = true;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::stringstream ss;
            ss << "    [" << i << "," << j << "]";
            all_pass &= compare_double(ss.str(), cpp_mat(i, j), py_ref[i * 2 + j]);
        }
    }

    return all_pass;
}

// ============================================================================
// Test 1: value()
// ============================================================================
void test_value() {
    std::cout << "\n=== Test 1: value() ===" << std::endl;

    // Create model
    auto model = std::make_shared<KinematicModel>();

    // Create obstacle
    Eigen::Matrix<double, 4, 1> obs_state;
    obs_state << 15.0, 2.0, 0.0, 0.0;  // [x, y, v, yaw]

    Eigen::Matrix<double, 3, 1> obs_attr;
    obs_attr << 2.0, 4.5, 1.5;  // [width, length, safety_buffer]

    auto obs = std::make_shared<Obstacle>(obs_state, obs_attr);
    std::vector<std::shared_ptr<Obstacle>> obstacles;
    obstacles.push_back(obs);

    // Create StateConstraint
    StateConstraint state_constraint(4, 2, model.get(), obstacles);

    // Test inputs
    int step = 10;
    Eigen::Matrix<double, 4, 1> state;
    state << 10.0, 0.0, 7.0, 0.0;  // [x, y, v, yaw]

    Eigen::Matrix<double, 2, 1> control;
    control << 0.5, 0.0;  // [acceleration, steering]

    double val;
    bool success = state_constraint.value(step, state, control, val);

    // Python reference value: 19.446378042722525
    double py_ref = 19.446378042722525;

    if (success) {
        compare_double("Cost value", val, py_ref);
    } else {
        std::cout << "  ERROR: value() returned false" << std::endl;
        ++tests_failed;
    }
}

// ============================================================================
// Test 2: gradient_lx()
// ============================================================================
void test_gradient_lx() {
    std::cout << "\n=== Test 2: gradient_lx() ===" << std::endl;

    // Create model
    auto model = std::make_shared<KinematicModel>();

    // Create obstacle
    Eigen::Matrix<double, 4, 1> obs_state;
    obs_state << 15.0, 2.0, 0.0, 0.0;

    Eigen::Matrix<double, 3, 1> obs_attr;
    obs_attr << 2.0, 4.5, 1.5;

    auto obs = std::make_shared<Obstacle>(obs_state, obs_attr);
    std::vector<std::shared_ptr<Obstacle>> obstacles;
    obstacles.push_back(obs);

    // Create StateConstraint
    StateConstraint state_constraint(4, 2, model.get(), obstacles);

    // Test inputs
    int step = 10;
    Eigen::Matrix<double, 4, 1> state;
    state << 10.0, 0.0, 7.0, 0.0;

    Eigen::Matrix<double, 2, 1> control;
    control << 0.5, 0.0;

    Eigen::Matrix<double, 4, 1> lx;
    bool success = state_constraint.gradient_lx(step, state, control, lx);

    // Python reference values
    double py_ref[4] = {
        31.721225099136213,
        36.511566604409943,
        0.000001019649056,
        65.70719770788460
    };

    if (success) {
        compare_vector_4d("Gradient lx", lx, py_ref);
    } else {
        std::cout << "  ERROR: gradient_lx() returned false" << std::endl;
        ++tests_failed;
    }
}

// ============================================================================
// Test 3: gradient_lu()
// ============================================================================
void test_gradient_lu() {
    std::cout << "\n=== Test 3: gradient_lu() ===" << std::endl;

    // Create model
    auto model = std::make_shared<KinematicModel>();

    // Create obstacle
    Eigen::Matrix<double, 4, 1> obs_state;
    obs_state << 15.0, 2.0, 0.0, 0.0;

    Eigen::Matrix<double, 3, 1> obs_attr;
    obs_attr << 2.0, 4.5, 1.5;

    auto obs = std::make_shared<Obstacle>(obs_state, obs_attr);
    std::vector<std::shared_ptr<Obstacle>> obstacles;
    obstacles.push_back(obs);

    // Create StateConstraint
    StateConstraint state_constraint(4, 2, model.get(), obstacles);

    // Test inputs
    int step = 10;
    Eigen::Matrix<double, 4, 1> state;
    state << 10.0, 0.0, 7.0, 0.0;

    Eigen::Matrix<double, 2, 1> control;
    control << 0.5, 0.0;

    Eigen::Matrix<double, 2, 1> lu;
    bool success = state_constraint.gradient_lu(step, state, control, lu);

    // Python reference values: all zeros (state constraints don't depend on control)
    double py_ref[2] = {0.0, 0.0};

    if (success) {
        compare_vector_2d("Gradient lu", lu, py_ref);
    } else {
        std::cout << "  ERROR: gradient_lu() returned false" << std::endl;
        ++tests_failed;
    }
}

// ============================================================================
// Test 4: hessian_lxx()
// ============================================================================
void test_hessian_lxx() {
    std::cout << "\n=== Test 4: hessian_lxx() ===" << std::endl;

    // Create model
    auto model = std::make_shared<KinematicModel>();

    // Create obstacle
    Eigen::Matrix<double, 4, 1> obs_state;
    obs_state << 15.0, 2.0, 0.0, 0.0;

    Eigen::Matrix<double, 3, 1> obs_attr;
    obs_attr << 2.0, 4.5, 1.5;

    auto obs = std::make_shared<Obstacle>(obs_state, obs_attr);
    std::vector<std::shared_ptr<Obstacle>> obstacles;
    obstacles.push_back(obs);

    // Create StateConstraint
    StateConstraint state_constraint(4, 2, model.get(), obstacles);

    // Test inputs
    int step = 10;
    Eigen::Matrix<double, 4, 1> state;
    state << 10.0, 0.0, 7.0, 0.0;

    Eigen::Matrix<double, 2, 1> control;
    control << 0.5, 0.0;

    Eigen::Matrix4d lxx;
    bool success = state_constraint.hessian_lxx(step, state, control, lxx);

    // Python reference values
    double py_ref[16] = {
        51.750925930410069, 59.558218553480231, 0.0, 107.157579901298703,
        59.558218553480231, 68.552329134810492, 0.0, 123.368616104599752,
        0.0, 0.0, 0.000005862982071, 0.0,
        107.157579901298703, 123.368616104599752, 0.0, 222.109546396786016
    };

    if (success) {
        compare_matrix_4x4("Hessian lxx", lxx, py_ref);
    } else {
        std::cout << "  ERROR: hessian_lxx() returned false" << std::endl;
        ++tests_failed;
    }
}

// ============================================================================
// Test 5: hessian_luu()
// ============================================================================
void test_hessian_luu() {
    std::cout << "\n=== Test 5: hessian_luu() ===" << std::endl;

    // Create model
    auto model = std::make_shared<KinematicModel>();

    // Create obstacle
    Eigen::Matrix<double, 4, 1> obs_state;
    obs_state << 15.0, 2.0, 0.0, 0.0;

    Eigen::Matrix<double, 3, 1> obs_attr;
    obs_attr << 2.0, 4.5, 1.5;

    auto obs = std::make_shared<Obstacle>(obs_state, obs_attr);
    std::vector<std::shared_ptr<Obstacle>> obstacles;
    obstacles.push_back(obs);

    // Create StateConstraint
    StateConstraint state_constraint(4, 2, model.get(), obstacles);

    // Test inputs
    int step = 10;
    Eigen::Matrix<double, 4, 1> state;
    state << 10.0, 0.0, 7.0, 0.0;

    Eigen::Matrix<double, 2, 1> control;
    control << 0.5, 0.0;

    Eigen::Matrix2d luu;
    bool success = state_constraint.hessian_luu(step, state, control, luu);

    // Python reference values: all zeros
    double py_ref[4] = {
        0.0, 0.0,
        0.0, 0.0
    };

    if (success) {
        compare_matrix_2x2("Hessian luu", luu, py_ref);
    } else {
        std::cout << "  ERROR: hessian_luu() returned false" << std::endl;
        ++tests_failed;
    }
}

// ============================================================================
// Test 6: hessian_lxu()
// ============================================================================
void test_hessian_lxu() {
    std::cout << "\n=== Test 6: hessian_lxu() ===" << std::endl;

    // Create model
    auto model = std::make_shared<KinematicModel>();

    // Create obstacle
    Eigen::Matrix<double, 4, 1> obs_state;
    obs_state << 15.0, 2.0, 0.0, 0.0;

    Eigen::Matrix<double, 3, 1> obs_attr;
    obs_attr << 2.0, 4.5, 1.5;

    auto obs = std::make_shared<Obstacle>(obs_state, obs_attr);
    std::vector<std::shared_ptr<Obstacle>> obstacles;
    obstacles.push_back(obs);

    // Create StateConstraint
    StateConstraint state_constraint(4, 2, model.get(), obstacles);

    // Test inputs
    int step = 10;
    Eigen::Matrix<double, 4, 1> state;
    state << 10.0, 0.0, 7.0, 0.0;

    Eigen::Matrix<double, 2, 1> control;
    control << 0.5, 0.0;

    Eigen::Matrix<double, 4, 2> lxu;
    bool success = state_constraint.hessian_lxu(step, state, control, lxu);

    // Python reference values: all zeros
    double py_ref[8] = {
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0
    };

    if (success) {
        compare_matrix_4x2("Hessian lxu", lxu, py_ref);
    } else {
        std::cout << "  ERROR: hessian_lxu() returned false" << std::endl;
        ++tests_failed;
    }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << std::fixed;
    std::cout << std::setprecision(15);

    std::cout << "============================================" << std::endl;
    std::cout << "StateConstraint C++ vs Python Verification" << std::endl;
    std::cout << "============================================" << std::endl;

    // Parse command line arguments
    int test_to_run = -1;
    if (argc > 1) {
        test_to_run = std::atoi(argv[1]);
    }

    if (test_to_run == -1 || test_to_run == 1) test_value();
    if (test_to_run == -1 || test_to_run == 2) test_gradient_lx();
    if (test_to_run == -1 || test_to_run == 3) test_gradient_lu();
    if (test_to_run == -1 || test_to_run == 4) test_hessian_lxx();
    if (test_to_run == -1 || test_to_run == 5) test_hessian_luu();
    if (test_to_run == -1 || test_to_run == 6) test_hessian_lxu();

    // Print summary
    std::cout << "\n============================================" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  Passed: " << GREEN << tests_passed << RESET << std::endl;
    std::cout << "  Failed: " << (tests_failed > 0 ? RED : "") << tests_failed << RESET << std::endl;
    std::cout << "============================================" << std::endl;

    return (tests_failed == 0) ? 0 : 1;
}
