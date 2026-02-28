/**
 * @file ilqr_demo.cpp
 * @brief Integration demo for iLQR trajectory optimization
 *
 * This demo replicates the scenario from scripts/test.py:
 * - Vehicle starting at [0, 0, 5.0, 0] (x, y, v, yaw)
 * - Reference path: straight line from (0,0) to (50,0)
 * - Two obstacles to avoid
 * - Outputs optimal trajectory as CSV
 */

#include "../include/ilqr_solver.hpp"
#include "../include/kinematic_model.hpp"
#include "../include/obstacle.hpp"
#include "../include/cost_calculator.hpp"
#include "../include/common_variable.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

using namespace ilqr;

// Type definitions using ILQR_PROBLEM_VARIABLES macro
ILQR_PROBLEM_VARIABLES(double, 4, 2)

int main(int argc, char* argv[]) {
    std::cout << "============================================" << std::endl;
    std::cout << "iLQR Trajectory Optimization Demo" << std::endl;
    std::cout << "============================================" << std::endl;

    // Parse command line arguments
    std::string output_file = "trajectory.csv";
    bool verbose = true;
    bool no_obstacles = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "-q") {
            verbose = false;
        } else if (arg == "--no-obstacles") {
            no_obstacles = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -o <file>         Output CSV file (default: trajectory.csv)" << std::endl;
            std::cout << "  -q                Quiet mode (minimal output)" << std::endl;
            std::cout << "  --no-obstacles    Run without obstacles (for testing)" << std::endl;
            std::cout << "  -h, --help        Show this help message" << std::endl;
            return 0;
        }
    }

    // ============================================================================
    // 1. Create kinematic model
    // ============================================================================
    if (verbose) {
        std::cout << "\n[1/6] Creating kinematic model..." << std::endl;
    }
    auto model = std::make_shared<KinematicModel>();

    // ============================================================================
    // 2. Create reference waypoints (2 x 1000)
    // ============================================================================
    if (verbose) {
        std::cout << "[2/6] Creating reference waypoints..." << std::endl;
    }
    int num_waypoints = 1000;
    Eigen::Matrix<double, 2, Eigen::Dynamic> ref_waypoints(2, num_waypoints);
    for (int i = 0; i < num_waypoints; ++i) {
        double x = 50.0 * i / (num_waypoints - 1);  // 0 to 50
        double y = 0.0;
        ref_waypoints(0, i) = x;
        ref_waypoints(1, i) = y;
    }
    if (verbose) {
        std::cout << "    Reference path: (0, 0) to (50, 0)" << std::endl;
        std::cout << "    Number of waypoints: " << num_waypoints << std::endl;
    }

    // ============================================================================
    // 3. Create obstacles (matching test.py)
    // ============================================================================
    if (verbose) {
        std::cout << "[3/6] Creating obstacles..." << std::endl;
    }
    std::vector<std::shared_ptr<Obstacle>> obstacles;

    if (!no_obstacles) {
        // Obstacle attributes: [width, length, safety_buffer]
        Eigen::Matrix<double, 3, 1> obs_attr;
        obs_attr << 2.0, 4.5, 1.5;  // Same as test.py

        // Obstacle 1: [x=5, y=-0.2, v=3.0, yaw=0.0]
        State obs_state_1;
        obs_state_1 << 6.5, -0.2, 3.0, 0.0;
        obstacles.push_back(std::make_shared<Obstacle>(obs_state_1, obs_attr));
        if (verbose) {
            std::cout << "    Obstacle 1: [x=5.0, y=-0.2, v=3.0, yaw=0.0]" << std::endl;
        }

        // Obstacle 2: [x=20, y=4, v=3.0, yaw=pi/2]
        State obs_state_2;
        obs_state_2 << 20.0, 4.0, 2.0, 0.0;
        obstacles.push_back(std::make_shared<Obstacle>(obs_state_2, obs_attr));
        if (verbose) {
            std::cout << "    Obstacle 2: [x=20.0, y=4.0, v=3.0, yaw=pi/2]" << std::endl;
        }
    } else {
        if (verbose) {
            std::cout << "    No obstacles (test mode)" << std::endl;
        }
    }

    // ============================================================================
    // 4. Create CostCalculator
    // ============================================================================
    if (verbose) {
        std::cout << "[4/6] Creating cost calculator..." << std::endl;
    }
    CostCalculator cost_calculator(model.get(), obstacles, ref_waypoints, 60);

    // ============================================================================
    // 5. Create ILQRSolver
    // ============================================================================
    if (verbose) {
        std::cout << "[5/6] Creating iLQR solver..." << std::endl;
    }
    ILQRSolver ilqr_solver(model.get(), &cost_calculator, 60);

    // ============================================================================
    // 6. Set initial state and solve
    // ============================================================================
    if (verbose) {
        std::cout << "[6/6] Running solver..." << std::endl;
        std::cout << "\nInitial state: [x=0.0, y=0.0, v=5.0, yaw=0.0]" << std::endl;
        std::cout << "Planning horizon: " << ilqr_solver.get_horizon() << " steps" << std::endl;
        std::cout << "Time step: " << ilqr_solver.get_dt() << "s" << std::endl;
        std::cout << "\n--- Solver iterations ---" << std::endl;
    }

    // Initial state: [x, y, v, yaw]
    State x0;
    x0 << 0.0, 0.0, 5.0, 0.0;

    // Solve
    Controls controls;
    States states;
    bool success = ilqr_solver.solve(x0, controls, states);

    if (!success) {
        std::cerr << "\nERROR: Solver failed to converge!" << std::endl;
        return 1;
    }

    // ============================================================================
    // 7. Output results
    // ============================================================================
    if (verbose) {
        std::cout << "\n============================================" << std::endl;
        std::cout << "Solver converged successfully!" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "\nTrajectory summary:" << std::endl;
        std::cout << "  Number of states: " << states.size() << std::endl;
        std::cout << "  Number of controls: " << controls.size() << std::endl;
        std::cout << "\nFinal state: ";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "[x=" << states.back()(0) << ", y=" << states.back()(1)
                  << ", v=" << states.back()(2) << ", yaw=" << states.back()(3) << "]" << std::endl;

        std::cout << "\nFirst few states:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), states.size()); ++i) {
            std::cout << "  x[" << i << "]: [" << states[i](0) << ", " << states[i](1)
                      << ", " << states[i](2) << ", " << states[i](3) << "]" << std::endl;
        }

        std::cout << "\nLast few states:" << std::endl;
        for (size_t i = std::max(size_t(0), states.size() - 5); i < states.size(); ++i) {
            std::cout << "  x[" << i << "]: [" << states[i](0) << ", " << states[i](1)
                      << ", " << states[i](2) << ", " << states[i](3) << "]" << std::endl;
        }
    }

    // Write trajectory to CSV file
    std::ofstream out_file(output_file);
    if (out_file.is_open()) {
        out_file << "# i, x, y, v, yaw" << std::endl;
        out_file << std::fixed << std::setprecision(8);
        for (size_t i = 0; i < states.size(); ++i) {
            out_file << i << ", "
                     << states[i](0) << ", "
                     << states[i](1) << ", "
                     << states[i](2) << ", "
                     << states[i](3) << std::endl;
        }
        out_file.close();
        std::cout << "\nTrajectory saved to: " << output_file << std::endl;
    } else {
        std::cerr << "Warning: Could not write to file: " << output_file << std::endl;
    }

    // Also output controls if verbose
    if (verbose) {
        std::cout << "\nFirst few controls:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), controls.size()); ++i) {
            std::cout << "  u[" << i << "]: [accel=" << controls[i](0)
                      << ", steer=" << controls[i](1) << "]" << std::endl;
        }

        std::cout << "\nLast few controls:" << std::endl;
        for (size_t i = std::max(size_t(0), controls.size() - 5); i < controls.size(); ++i) {
            std::cout << "  u[" << i << "]: [accel=" << controls[i](0)
                      << ", steer=" << controls[i](1) << "]" << std::endl;
        }
    }

    return 0;
}
