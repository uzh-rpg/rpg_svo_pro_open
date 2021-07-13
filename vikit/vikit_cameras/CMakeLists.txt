project(vikit_cameras)
cmake_minimum_required (VERSION 2.8.3)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

include(SvoSetup)

set(HEADERS
  include/vikit/cameras.h
  include/vikit/cameras/camera_geometry_base.h
  include/vikit/cameras/implementation/camera_geometry_base.hpp
  include/vikit/cameras/no_distortion.h
  include/vikit/cameras/atan_distortion.h
  include/vikit/cameras/equidistant_distortion.h
  include/vikit/cameras/equidistant_fisheye_geometry.h
  include/vikit/cameras/equidistant_fisheye_projection.h
  include/vikit/cameras/omni_geometry.h
  include/vikit/cameras/omni_projection.h
  include/vikit/cameras/radial_tangential_distortion.h
  include/vikit/cameras/pinhole_projection.h
  include/vikit/cameras/implementation/pinhole_projection.hpp
  include/vikit/cameras/camera_geometry.h
  include/vikit/cameras/implementation/camera_geometry.hpp
  include/vikit/cameras/camera_factory.h
  include/vikit/cameras/ncamera.h
  include/vikit/cameras/yaml/camera-yaml-serialization.h
  include/vikit/cameras/yaml/ncamera-yaml-serialization.h
  )

set(SOURCES
  src/camera_factory.cpp
  src/ncamera.cpp
  src/camera_geometry_base.cpp
  src/camera_yaml_serialization.cpp
  src/equidistant_fisheye_geometry.cpp
  src/equidistant_fisheye_projection.cpp
  src/omni_geometry.cpp
  src/omni_projection.cpp
  src/ncamera_yaml_serialization.cpp
  )

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} yaml-cpp)

################################################################################
# Tests

catkin_add_gtest(${PROJECT_NAME}_tests
  test/test_cameras
)
target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME})

################################################################################
# Install

cs_install()
cs_export()
