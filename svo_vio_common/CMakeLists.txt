project(svo_vio_common)
cmake_minimum_required(VERSION 2.8.3)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

list(APPEND catkin_LIBRARIES)

#############
# LIBRARIES #
#############
set(HEADERS
  include/svo/vio_common/logging.hpp
  include/svo/vio_common/matrix_operations.hpp
  include/svo/vio_common/matrix.hpp
  include/svo/vio_common/test_utils.hpp
  include/svo/vio_common/backend_types.hpp
  )

set(SOURCES
  src/vio_common.cpp
  src/test_utils.cpp
  )

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})


##########
# GTESTS #
##########


##########
# EXPORT #
##########
cs_install()
cs_export()
