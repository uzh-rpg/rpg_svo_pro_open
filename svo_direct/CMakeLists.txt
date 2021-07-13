project(svo_direct)
cmake_minimum_required(VERSION 2.8.3)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
find_package(OpenCV REQUIRED)

include(SvoSetup)

set(HEADERS
  include/svo/direct/patch_warp.h
  include/svo/direct/patch_score.h
  include/svo/direct/patch_utils.h
  include/svo/direct/feature_alignment.h
  include/svo/direct/feature_detection_types.h
  include/svo/direct/feature_detection.h
  include/svo/direct/feature_detection_utils.h
  include/svo/direct/matcher.h
  include/svo/direct/depth_filter.h
  include/svo/direct/depth_estimation.h
  include/svo/direct/elder_zucker.h
)

set(SOURCES
  src/patch_warp.cpp
  src/feature_alignment.cpp
  src/feature_detection.cpp
  src/feature_detection_utils.cpp
  src/matcher.cpp
  src/depth_filter.cpp
  src/depth_estimation.cpp
  src/elder_zucker.cpp
  )

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

cs_add_executable(svo_test_patch_score test/test_patch_score.cpp)
target_link_libraries(svo_test_patch_score ${PROJECT_NAME})

cs_add_executable(svo_test_edgelet_direction test/test_edgelet_direction.cpp)
target_link_libraries(svo_test_edgelet_direction ${PROJECT_NAME})

cs_add_executable(svo_test_edgelet_extraction test/test_edgelet_extraction.cpp)
target_link_libraries(svo_test_edgelet_extraction ${PROJECT_NAME})

cs_add_executable(svo_test_feature_alignment test/test_feature_alignment.cpp)
target_link_libraries(svo_test_feature_alignment ${PROJECT_NAME})

################################################################################
###
### GTEST
###

catkin_add_gtest(${PROJECT_NAME}-test-depth-estimation
  test/test_depth_estimation.cpp
  test/test_utils.h
)
target_link_libraries(${PROJECT_NAME}-test-depth-estimation ${PROJECT_NAME} pthread)


################################################################################
cs_install()
cs_export()

