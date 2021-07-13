project(svo_ceres_backend)
cmake_minimum_required(VERSION 2.8.3)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

################################################################################
# generic setup (compiler flags, etc.)
include(SvoSetup)

# OpenMP
find_package(OpenMP QUIET)
if (OPENMP_FOUND)
  message(STATUS "OpenMP found")
  add_definitions(-DUSE_OPENMP)
endif ()

#############
# LIBRARIES #
#############
set(HEADERS
  include/svo/ceres_backend_interface.hpp
  include/svo/ceres_backend/estimator.hpp
  include/svo/ceres_backend/estimator_impl.hpp
  include/svo/ceres_backend/estimator_types.hpp
  include/svo/ceres_backend/ceres_iteration_callback.hpp
  include/svo/ceres_backend/error_interface.hpp
  include/svo/ceres_backend/homogeneous_point_error.hpp
  include/svo/ceres_backend/homogeneous_point_local_parameterization.hpp
  include/svo/ceres_backend/homogeneous_point_parameter_block.hpp
  include/svo/ceres_backend/imu_error.hpp
  include/svo/ceres_backend/local_parameterization_additional_interfaces.hpp
  include/svo/ceres_backend/map.hpp
  include/svo/ceres_backend/marginalization_error.hpp
  include/svo/ceres_backend/marginalization_error_impl.hpp
  include/svo/ceres_backend/parameter_block.hpp
  include/svo/ceres_backend/pose_error.hpp
  include/svo/ceres_backend/pose_local_parameterization.hpp
  include/svo/ceres_backend/pose_parameter_block.hpp
  include/svo/ceres_backend/relative_pose_error.hpp
  include/svo/ceres_backend/reprojection_error.hpp
  include/svo/ceres_backend/reprojection_error_impl.hpp
  include/svo/ceres_backend/reprojection_error_base.hpp
  include/svo/ceres_backend/speed_and_bias_error.hpp
  include/svo/ceres_backend/speed_and_bias_parameter_block.hpp
  include/svo/ceres_backend/map_alignment.hpp
  include/svo/ceres_backend/epipolar_error.hpp
  include/svo/ceres_backend/reprojection_error_simple.hpp
  include/svo/motion_detector.hpp
  include/svo/outlier_rejection.hpp
  include/svo/ceres_backend_publisher.hpp
  )

set(SOURCES
  src/ceres_backend_interface.cpp
  src/estimator.cpp
  src/error_interface.cpp
  src/homogeneous_point_error.cpp
  src/homogeneous_point_parameter_block.cpp
  src/homogeneous_point_local_parameterization.cpp
  src/imu_error.cpp
  src/local_parameterization_additional_interfaces.cpp
  src/map.cpp
  src/marginalization_error.cpp
  src/pose_error.cpp
  src/pose_parameter_block.cpp
  src/pose_local_parameterization.cpp
  src/relative_pose_error.cpp
  src/speed_and_bias_error.cpp
  src/speed_and_bias_parameter_block.cpp
  src/motion_detector.cpp
  src/outlier_rejection.cpp
  src/ceres_backend_publisher.cpp
  src/map_alignment.cpp
  )

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

##########
# GTESTS #
##########
catkin_add_gtest(test_backend_id test/test_backend_id.cpp)
target_link_libraries(test_backend_id ${PROJECT_NAME})

catkin_add_gtest(test_estimator test/test_estimator.cpp)
target_link_libraries(test_estimator ${PROJECT_NAME})

catkin_add_gtest(test_homogeneous_point_error test/test_homogeneous_point_error.cpp)
target_link_libraries(test_homogeneous_point_error ${PROJECT_NAME})

catkin_add_gtest(test_map test/test_map.cpp)
target_link_libraries(test_map ${PROJECT_NAME})

catkin_add_gtest(test_marginalization test/test_marginalization.cpp)
target_link_libraries(test_marginalization ${PROJECT_NAME})

catkin_add_gtest(test_pose_error test/test_pose_error.cpp)
target_link_libraries(test_pose_error ${PROJECT_NAME})

catkin_add_gtest(test_pose_local_parameterization test/test_pose_local_parameterization.cpp)
target_link_libraries(test_pose_local_parameterization ${PROJECT_NAME})

catkin_add_gtest(test_reprojection_error test/test_reprojection_error.cpp)
target_link_libraries(test_reprojection_error ${PROJECT_NAME})

catkin_add_gtest(test_relative_pose_error test/test_relative_pose_error.cpp)
target_link_libraries(test_relative_pose_error ${PROJECT_NAME})

catkin_add_gtest(test_speed_and_bias_error test/test_speed_and_bias_error.cpp)
target_link_libraries(test_speed_and_bias_error ${PROJECT_NAME})

catkin_add_gtest(test_imu_error test/test_imu_error.cpp)
target_link_libraries(test_imu_error ${PROJECT_NAME})

##########
# EXPORT #
##########
cs_install()
cs_export()
