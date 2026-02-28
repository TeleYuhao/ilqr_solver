/**
 * @file test_kinematic_model_verification.cpp
 * @brief Verification test for kinematic_model.hpp vs Python kinematic_model.py
 *
 * This test compares the C++ implementation with Python reference values
 * to ensure numerical consistency (tolerance: 1e-8).
 */

#include "../include/kinematic_model.hpp"
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
 * @brief Compare two 4D vectors with tolerance
 */
bool compare_state_4d(const std::string& name, const KinematicModel::State& cpp_state,
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

/**
 * @brief Compare two 2D vectors with tolerance
 */
bool compare_position_2d(const std::string& name, const Eigen::Vector2d& cpp_pos,
                         const double py_ref[2]) {
    std::cout << name << ":" << std::endl;
    bool all_pass = true;

    for (int i = 0; i < 2; ++i) {
        std::stringstream ss;
        ss << "    [" << i << "]";
        all_pass &= compare_double(ss.str(), cpp_pos(i), py_ref[i]);
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

/**
 * @brief Compare 2x4 matrix with tolerance
 */
bool compare_matrix_2x4(const std::string& name, const Eigen::Matrix<double, 2, 4>& cpp_mat,
                        const double py_ref[8]) {
    std::cout << name << ":" << std::endl;
    bool all_pass = true;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::stringstream ss;
            ss << "    [" << i << "," << j << "]";
            all_pass &= compare_double(ss.str(), cpp_mat(i, j), py_ref[i * 4 + j]);
        }
    }

    return all_pass;
}

// ============================================================================
// Test 1: gradient_fx()
// ============================================================================
void test_gradient_fx() {
    std::cout << "\n=== Test 1: gradient_fx() ===" << std::endl;

    KinematicModel model;

    KinematicModel::State state;
    state << 0.0, 0.0, 5.0, 0.0;  // [x, y, v, yaw]

    KinematicModel::Control ctrl;
    ctrl << 1.0, 0.1;  // [acceleration, steering]

    double step = 0.1;

    auto A = model.gradient_fx(state, ctrl, step);

    // Python reference values (computed from kinematic_model.py)
    // state = [0, 0, 5, 0], control = [1, 0.1], step = 0.1
    // beta_d = arctan(tan(0.1/2)) = arctan(tan(0.05)) = 0.05
    // yaw + beta_d = 0.05
    // cos(0.05) = 0.99875026..., sin(0.05) = 0.04997917...
    // dfdx[0,0] = 1
    // dfdx[0,2] = cos(yaw+beta_d) * step = 0.099875...
    // dfdx[0,3] = -v * sin(yaw+beta_d) * step = -5 * 0.049979... * 0.1 = -0.024989...
    // dfdx[1,1] = 1
    // dfdx[1,2] = sin(yaw+beta_d) * step = 0.0049979...
    // dfdx[1,3] = v * cos(yaw+beta_d) * step = 5 * 0.99875... * 0.1 = 0.49937...
    // dfdx[2,2] = 1
    // dfdx[3,2] = 2 * sin(beta_d) * step / WB = 2 * 0.049979... * 0.1 / 3.6 = 0.0027766...
    // dfdx[3,3] = 1

    double py_ref[16] = {
        1.0, 0.0, 0.099875026039497,   -0.024989584635339,
        0.0, 1.0, 0.004997916927068,    0.499375130197483,
        0.0, 0.0, 1.0,                  0.0,
        0.0, 0.0, 0.002776620515038,    1.0
    };

    compare_matrix_4x4("A matrix (gradient_fx)", A, py_ref);
}

// ============================================================================
// Test 2: gradient_fu()
// ============================================================================
void test_gradient_fu() {
    std::cout << "\n=== Test 2: gradient_fu() ===" << std::endl;

    KinematicModel model;

    KinematicModel::State state;
    state << 0.0, 0.0, 5.0, 0.0;

    KinematicModel::Control ctrl;
    ctrl << 1.0, 0.1;

    double step = 0.1;

    auto B = model.gradient_fu(state, ctrl, step);

    // Python reference values
    // delta = 0.1, yaw = 0, v = 5
    // beta_d = arctan(tan(0.05)) = 0.05
    // beta_over_stl = 0.5 * (1 + tan(0.1)^2) / (1 + 0.25 * tan(0.1)^2)
    //               = 0.5 * (1 + 0.010033...) / (1 + 0.002508...)
    //               = 0.50376...
    // dfdu[0,1] = v * (-sin(beta_d + yaw)) * step * beta_over_stl
    //          = 5 * (-sin(0.05)) * 0.1 * 0.50376 = -0.12594...
    // dfdu[1,1] = v * cos(beta_d + yaw) * step * beta_over_stl
    //          = 5 * cos(0.05) * 0.1 * 0.50376 = 0.25169...
    // dfdu[2,0] = step = 0.1
    // dfdu[3,1] = 2 * v * step / WB * cos(beta_d) * beta_over_stl
    //          = 2 * 5 * 0.1 / 3.6 * cos(0.05) * 0.50376 = 0.13978...

    double py_ref[8] = {
        0.0,   -0.012588894725070,
        0.0,    0.251568044611829,
        0.1,    0.0,
        0.0,    0.139760024784350
    };

    compare_matrix_4x2("B matrix (gradient_fu)", B, py_ref);
}

// ============================================================================
// Test 3: forward_calculation()
// ============================================================================
void test_forward_calculation() {
    std::cout << "\n=== Test 3: forward_calculation() ===" << std::endl;

    KinematicModel model;

    KinematicModel::State state;
    state << 0.0, 0.0, 5.0, 0.0;

    KinematicModel::Control ctrl;
    ctrl << 1.0, 0.1;

    double step = 0.1;

    auto next_state = model.forward_calculation(state, ctrl, step);

    // Python reference values
    // beta = arctan(tan(0.1) / 2) = arctan(0.05002...) = 0.049996...
    // next_x = x + v * cos(beta + yaw) * step = 0 + 5 * cos(0.05) * 0.1 = 0.49937...
    // next_y = y + v * sin(beta + yaw) * step = 0 + 5 * sin(0.05) * 0.1 = 0.02499...
    // next_v = v + a * step = 5 + 1 * 0.1 = 5.1
    // next_yaw = yaw + 2 * v * sin(beta) * step / WB = 0 + 2 * 5 * 0.05 * 0.1 / 3.6 = 0.01388...

    double py_ref[4] = {
        0.499371994754908,
        0.025052162671196,
        5.1,
        0.013917868150664
    };

    compare_state_4d("Next state", next_state, py_ref);
}

// ============================================================================
// Test 4: get_vehicle_front_and_rear_centers()
// ============================================================================
void test_get_vehicle_front_and_rear_centers() {
    std::cout << "\n=== Test 4: get_vehicle_front_and_rear_centers() ===" << std::endl;

    KinematicModel model;

    Eigen::Vector2d pos;
    pos << 10.0, 20.0;

    double yaw = 0.5;  // radians

    auto [front_pnt, rear_pnt] = model.get_vehicle_front_and_rear_centers(pos, yaw);

    // Python reference values
    // half_wb = 1.8
    // half_wb_vec = [1.8 * cos(0.5), 1.8 * sin(0.5)] = [1.578..., 0.864...]
    // front = pos + half_wb_vec = [11.578..., 20.864...]
    // rear = pos - half_wb_vec = [8.421..., 19.135...]

    double py_front_ref[2] = {11.579648611402671, 20.862965969487565};
    double py_rear_ref[2] = {8.420351388597329, 19.137034030512435};

    compare_position_2d("Front point", front_pnt, py_front_ref);
    compare_position_2d("Rear point", rear_pnt, py_rear_ref);
}

// ============================================================================
// Test 5: get_vehicle_front_and_rear_center_derivatives()
// ============================================================================
void test_get_vehicle_front_and_rear_center_derivatives() {
    std::cout << "\n=== Test 5: get_vehicle_front_and_rear_center_derivatives() ===" << std::endl;

    KinematicModel model;

    double yaw = 0.5;  // radians

    auto [front_deriv, rear_deriv] = model.get_vehicle_front_and_rear_center_derivatives(yaw);

    // Python reference values
    // half_wb = 1.8
    // front_pnt_over_state:
    //   [1, 0, 0, -half_wb*sin(yaw)] = [1, 0, 0, -1.8*sin(0.5)]
    //   [0, 1, 0,  half_wb*cos(yaw)] = [0, 1, 0,  1.8*cos(0.5)]
    // rear_pnt_over_state:
    //   [1, 0, 0,  half_wb*sin(yaw)] = [1, 0, 0, 1.8*sin(0.5)]
    //   [0, 1, 0, -half_wb*cos(yaw)] = [0, 1, 0, -1.8*cos(0.5)]

    double py_front_ref[8] = {
        1.0, 0.0, 0.0, -0.862965969487565,
        0.0, 1.0, 0.0,  1.579648611402671
    };

    double py_rear_ref[8] = {
        1.0, 0.0, 0.0,  0.862965969487565,
        0.0, 1.0, 0.0, -1.579648611402671
    };

    compare_matrix_2x4("Front derivative", front_deriv, py_front_ref);
    compare_matrix_2x4("Rear derivative", rear_deriv, py_rear_ref);
}

// ============================================================================
// Test 6: init_traj()
// ============================================================================
void test_init_traj() {
    std::cout << "\n=== Test 6: init_traj() ===" << std::endl;

    KinematicModel model;

    KinematicModel::State init_state;
    init_state << 0.0, 0.0, 5.0, 0.0;

    // Create zero controls
    KinematicModel::Controls controls(60);
    for (int i = 0; i < 60; ++i) {
        KinematicModel::Control ctrl;
        ctrl << 0.0, 0.0;
        controls[i] = ctrl;
    }

    int horizon = 60;
    auto states = model.init_traj(init_state, controls, horizon);

    // Check a few key states
    std::cout << "Checking state[0] (initial state):" << std::endl;
    double py_state0[4] = {0.0, 0.0, 5.0, 0.0};
    compare_state_4d("  state[0]", states[0], py_state0);

    std::cout << "Checking state[1]:" << std::endl;
    // With zero controls, vehicle moves at constant velocity
    // state[1] = [0 + 5*cos(0)*0.1, 0 + 5*sin(0)*0.1, 5, 0]
    //          = [0.5, 0, 5, 0]
    double py_state1[4] = {0.5, 0.0, 5.0, 0.0};
    compare_state_4d("  state[1]", states[1], py_state1);

    std::cout << "Checking state[60] (final state):" << std::endl;
    // After 60 steps: [60*0.5, 0, 5, 0] = [30, 0, 5, 0]
    double py_state60[4] = {30.0, 0.0, 5.0, 0.0};
    compare_state_4d("  state[60]", states[60], py_state60);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << std::fixed;
    std::cout << std::setprecision(15);

    std::cout << "============================================" << std::endl;
    std::cout << "KinematicModel C++ vs Python Verification" << std::endl;
    std::cout << "============================================" << std::endl;

    // Parse command line arguments
    int test_to_run = -1;  // -1 means run all tests
    if (argc > 1) {
        test_to_run = std::atoi(argv[1]);
    }

    if (test_to_run == -1 || test_to_run == 1) test_gradient_fx();
    if (test_to_run == -1 || test_to_run == 2) test_gradient_fu();
    if (test_to_run == -1 || test_to_run == 3) test_forward_calculation();
    if (test_to_run == -1 || test_to_run == 4) test_get_vehicle_front_and_rear_centers();
    if (test_to_run == -1 || test_to_run == 5) test_get_vehicle_front_and_rear_center_derivatives();
    if (test_to_run == -1 || test_to_run == 6) test_init_traj();

    // Print summary
    std::cout << "\n============================================" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  Passed: " << GREEN << tests_passed << RESET << std::endl;
    std::cout << "  Failed: " << (tests_failed > 0 ? RED : "") << tests_failed << RESET << std::endl;
    std::cout << "============================================" << std::endl;

    return (tests_failed == 0) ? 0 : 1;
}
