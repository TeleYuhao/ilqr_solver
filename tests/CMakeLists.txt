# CMakeLists.txt for verification tests

cmake_minimum_required(VERSION 3.10)
project(VerificationTests)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Test executables
add_executable(test_kinematic_model_verification test_kinematic_model_verification.cpp)
add_executable(test_obstacle_verification test_obstacle_verification.cpp)
add_executable(test_state_cost_verification test_state_cost_verification.cpp)
add_executable(test_control_constraint_verification test_control_constraint_verification.cpp)
add_executable(test_state_constraint_verification test_state_constraint_verification.cpp)
add_executable(test_cost_calculator_verification test_cost_calculator_verification.cpp)
add_executable(test_ilqr_solver_verification test_ilqr_solver_verification.cpp)

# Link Eigen3
target_link_libraries(test_kinematic_model_verification ${EIGEN3_LIBS})
target_link_libraries(test_obstacle_verification ${EIGEN3_LIBS})
target_link_libraries(test_state_cost_verification ${EIGEN3_LIBS})
target_link_libraries(test_control_constraint_verification ${EIGEN3_LIBS})
target_link_libraries(test_state_constraint_verification ${EIGEN3_LIBS})
target_link_libraries(test_cost_calculator_verification ${EIGEN3_LIBS})
target_link_libraries(test_ilqr_solver_verification ${EIGEN3_LIBS})
