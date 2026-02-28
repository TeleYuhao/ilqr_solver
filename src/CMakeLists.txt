cmake_minimum_required(VERSION 3.10)
project(ILQR_Demos)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Demo executables
add_executable(ilqr_demo ilqr_demo.cpp)

# Link Eigen3
target_link_libraries(ilqr_demo ${EIGEN3_LIBS})
