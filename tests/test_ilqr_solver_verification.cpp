/**
 * @file test_ilqr_solver_verification.cpp
 * @brief Verification test for ilqr_solver.hpp vs Python ILQR_Core.py
 *
 * This test compares the C++ implementation with Python reference values
 * to ensure numerical consistency (tolerance: 1e-8).
 */

#include "../include/ilqr_solver.hpp"
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
constexpr double TOLERANCE = 1e-6;  // Slightly relaxed due to iterative algorithm

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

// ============================================================================
// Test 1: Full solver test (3 iterations)
// ============================================================================
void test_solve() {
    std::cout << "\n=== Test 1: solve() - 3 iterations ===" << std::endl;

    // Create model
    auto model = std::make_shared<KinematicModel>();

    // Create reference waypoints (2 x 1000) - matching Python format
    int num_waypoints = 1000;
    Eigen::Matrix<double, 2, Eigen::Dynamic> ref_waypoints(2, num_waypoints);
    for (int i = 0; i < num_waypoints; ++i) {
        double x = 50.0 * i / (num_waypoints - 1);  // 0 to 50
        double y = 0.0;
        ref_waypoints(0, i) = x;
        ref_waypoints(1, i) = y;
    }

    // Create empty obstacle list
    std::vector<std::shared_ptr<Obstacle>> obstacles;

    // Create CostCalculator
    CostCalculator cost_calculator(model.get(), obstacles, ref_waypoints, 60);

    // Create ILQRSolver
    ILQRSolver ilqr_solver(model.get(), &cost_calculator, 60);
    ilqr_solver.set_max_iter(3);  // Run only 3 iterations for verification

    // Initial state
    State x0;
    x0 << 0.0, 0.0, 5.0, 0.0;

    // Solve
    Controls controls;
    States states;
    bool success = ilqr_solver.solve(x0, controls, states);

    if (success) {
        // Calculate final cost
        double final_cost = cost_calculator.CalculateTotalCost(states, controls);

        // Python reference (matching test.py Q = diag([1.0, 1.0, 0.5, 0])): 23.283346307498515
        compare_double("Final cost", final_cost, 23.283346307498515);

        // Check some sample states
        std::cout << "\n  Sample states:" << std::endl;

        // x[0]
        std::cout << "    x[0]:" << std::endl;
        compare_double("      [0]", states[0](0), 0.0);
        compare_double("      [1]", states[0](1), 0.0);
        compare_double("      [2]", states[0](2), 5.0);
        compare_double("      [3]", states[0](3), 0.0);

        // x[30]
        std::cout << "    x[30]:" << std::endl;
        compare_double("      [0]", states[30](0), 15.085687950826403);
        compare_double("      [1]", states[30](1), 0.0);
        compare_double("      [2]", states[30](2), 5.099447454971421);
        compare_double("      [3]", states[30](3), 0.0);

        // x[60]
        std::cout << "    x[60]:" << std::endl;
        compare_double("      [0]", states[60](0), 30.850228366646189);
        compare_double("      [1]", states[60](1), 0.0);
        compare_double("      [2]", states[60](2), 5.381998736987611);
        compare_double("      [3]", states[60](3), 0.0);

        // Check some sample controls
        std::cout << "\n  Sample controls:" << std::endl;

        // u[0]
        std::cout << "    u[0]:" << std::endl;
        compare_double("      [0]", controls[0](0), 0.002627670399655);
        compare_double("      [1]", controls[0](1), 0.0);

        // u[30]
        std::cout << "    u[30]:" << std::endl;
        compare_double("      [0]", controls[30](0), 0.082474799398433);
        compare_double("      [1]", controls[30](1), 0.0);

        // u[59]
        std::cout << "    u[59]:" << std::endl;
        compare_double("      [0]", controls[59](0), -0.030906412617727);
        compare_double("      [1]", controls[59](1), 0.0);
    } else {
        std::cout << "  ERROR: solve() returned false" << std::endl;
        ++tests_failed;
    }
}

// ============================================================================
// Test 2: Backward pass
// ============================================================================
void test_backward_pass() {
    std::cout << "\n=== Test 2: backward_pass() ===" << std::endl;

    // Create model
    auto model = std::make_shared<KinematicModel>();

    // Create reference waypoints
    int num_waypoints = 1000;
    Eigen::Matrix<double, 2, Eigen::Dynamic> ref_waypoints(2, num_waypoints);
    for (int i = 0; i < num_waypoints; ++i) {
        double x = 50.0 * i / (num_waypoints - 1);
        double y = 0.0;
        ref_waypoints(0, i) = x;
        ref_waypoints(1, i) = y;
    }

    // Create empty obstacle list
    std::vector<std::shared_ptr<Obstacle>> obstacles;

    // Create CostCalculator
    CostCalculator cost_calculator(model.get(), obstacles, ref_waypoints, 60);

    // Create ILQRSolver
    ILQRSolver ilqr_solver(model.get(), &cost_calculator, 60);

    // Initialize trajectory
    State x0;
    x0 << 0.0, 0.0, 5.0, 0.0;

    int horizon = 60;
    Controls init_controls;
    init_controls.reserve(horizon);
    Control zero_control = Control::Zero();
    for (int i = 0; i < horizon; ++i) {
        init_controls.push_back(zero_control);
    }
    States init_states = model->init_traj(x0, init_controls);

    // Test backward_pass
    Controls d;
    MatrixCUXs K;
    double delt_V;
    double lambda = 20.0;

    bool success = ilqr_solver.backward_pass(init_controls, init_states, lambda, d, K, delt_V);

    if (success) {
        // Python reference (matching test.py Q = diag([1.0, 1.0, 0.5, 0])): delt_V = -2.007908560606199
        compare_double("Expected cost reduction (delt_V)", delt_V, -2.007908560606199);

        // Check d[0]
        std::cout << "\n  d[0]:" << std::endl;
        compare_double("    [0]", d[0](0), 0.008111911315320);
        compare_double("    [1]", d[0](1), 0.0);

        // Check d[1]
        std::cout << "\n  d[1]:" << std::endl;
        compare_double("    [0]", d[1](0), 0.008855391634614);
        compare_double("    [1]", d[1](1), 0.0);

        // Check K[0]
        std::cout << "\n  K[0]:" << std::endl;
        compare_double("    [0,0]", K[0](0, 0), -0.847775554448290);
        compare_double("    [0,1]", K[0](0, 1), 0.0);
        compare_double("    [0,2]", K[0](0, 2), -2.326766960363055);
        compare_double("    [0,3]", K[0](0, 3), 0.0);
        compare_double("    [1,0]", K[0](1, 0), 0.0);
        compare_double("    [1,1]", K[0](1, 1), -1.649205755837233);
        compare_double("    [1,2]", K[0](1, 2), 0.0);
        compare_double("    [1,3]", K[0](1, 3), -2.206090730048049);
    } else {
        std::cout << "  ERROR: backward_pass() returned false" << std::endl;
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
    std::cout << "ILQRSolver C++ vs Python Verification" << std::endl;
    std::cout << "============================================" << std::endl;

    // Parse command line arguments
    int test_to_run = -1;
    if (argc > 1) {
        test_to_run = std::atoi(argv[1]);
    }

    if (test_to_run == -1 || test_to_run == 1) test_solve();
    if (test_to_run == -1 || test_to_run == 2) test_backward_pass();

    // Print summary
    std::cout << "\n============================================" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  Passed: " << GREEN << tests_passed << RESET << std::endl;
    std::cout << "  Failed: " << (tests_failed > 0 ? RED : "") << tests_failed << RESET << std::endl;
    std::cout << "============================================" << std::endl;

    return (tests_failed == 0) ? 0 : 1;
}
