project(vikit_common)
cmake_minimum_required(VERSION 2.8.3)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
find_package(OpenCV REQUIRED)

include(SvoSetup)

set(HEADERS
  include/vikit/blender_utils.h
  include/vikit/math_utils.h
  include/vikit/performance_monitor.h
  include/vikit/ringbuffer.h
  include/vikit/sample.h
  include/vikit/timer.h
  include/vikit/user_input_thread.h
  include/vikit/vision.h
  include/vikit/homography_decomp.h
  include/vikit/homography.h
  include/vikit/path_utils.h
  include/vikit/csv_utils.h
  include/aslam/common/entrypoint.h
  include/aslam/common/macros.h
  include/aslam/common/memory.h
  include/aslam/common/pose-types.h
  include/aslam/common/yaml-serialization.h
  include/aslam/common/yaml-serialization-eigen.h
  )

set(SOURCES
  src/homography.cpp
  src/math_utils.cpp
  src/performance_monitor.cpp
  src/sample.cpp
  src/user_input_thread.cpp
  src/vision.cpp
  )

# Create vikit shared library
cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
################################################################################
# Unit tests
catkin_add_gtest(homography test/test_homography.cpp)
target_link_libraries(homography gtest ${PROJECT_NAME})

catkin_add_gtest(math_utils test/test_math_utils.cpp)
target_link_libraries(math_utils gtest ${PROJECT_NAME})

################################################################################
# Other tests
cs_add_executable(vikit_test_timer test/test_timer.cpp)
target_link_libraries(vikit_test_timer ${PROJECT_NAME})

cs_add_executable(vikit_test_common test/test_common.cpp)
target_link_libraries(vikit_test_common ${PROJECT_NAME})

################################################################################
cs_install()
cs_export()
