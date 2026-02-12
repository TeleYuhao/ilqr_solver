#pragma once

namespace ilqr::config {

// Vehicle parameters
constexpr double WHEELBASE = 3.6;   // meters
constexpr double WIDTH = 2.0;        // meters
constexpr double LENGTH = 4.5;       // meters
constexpr double EGO_WIDTH = 2.0;
constexpr double EGO_PNT_RADIUS = 1.0;
constexpr double SAFETY_BUFFER = 1.5;  // meters (matching Python)

// State and control bounds
constexpr double V_MAX = 10.0;
constexpr double V_MIN = 0.0;
constexpr double A_MAX = 2.0;
constexpr double A_MIN = -2.0;
constexpr double DELTA_MAX = 1.57;
constexpr double DELTA_MIN = -1.57;

// iLQR algorithm parameters
constexpr int HORIZON_LENGTH = 60;
constexpr double DT = 0.1;
constexpr double REF_VELO = 6.0;
constexpr double EXP_Q1 = 5.5;
constexpr double EXP_Q2 = 5.75;

// Solver hyperparameters
constexpr int MAX_ITER = 50;
constexpr double INIT_LAMB = 20.0;
constexpr double LAMB_DECAY = 0.7;
constexpr double LAMB_AMPLIFY = 2.0;
constexpr double MAX_LAMB = 10000.0;
constexpr double TOL = 0.001;

} // namespace ilqr::config
