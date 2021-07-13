cmake_minimum_required(VERSION 2.8.3)
project(vikit_solver)
find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")

#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME}
  include/vikit/solver/mini_least_squares_solver.h
  include/vikit/solver/implementation/mini_least_squares_solver.hpp
  src/robust_cost.cpp
  include/vikit/solver/robust_cost.h
)

## Declare a cpp executable
#cs_add_executable(elevation_map_node src/elevation_map_node.cpp)
#target_link_libraries(elevation_map_node ${PROJECT_NAME})

##########
# GTESTS #
##########
#catkin_add_gtest(test_cameras
#    test/test-cameras.cc
#)
#target_link_libraries(test_cameras ${PROJECT_NAME}) 

##########
# EXPORT #
##########
cs_install()
cs_export()
