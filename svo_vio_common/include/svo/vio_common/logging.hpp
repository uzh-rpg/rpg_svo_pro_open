// Copyright (C) ETH Zurich, Wyss Zurich, Zurich Eye - All Rights Reserved
// Unauthorized copying of this file, via any medium is strictly prohibited
// Proprietary and confidential

#pragma once

#pragma diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
// glog has an unused typedef.
// https://github.com/google/glog/pull/33
#include <glog/logging.h>
#pragma diagnostic pop

//! @file logging.hpp
//! Includes Glog framework and defines macros for DEBUG_CHECK_* which
//! can be compiled away.

#define DEBUG_CHECK(val) CHECK(val)
#define DEBUG_CHECK_NOTNULL(val) CHECK_NOTNULL(val)
#define DEBUG_CHECK_EQ(val1, val2) CHECK_EQ(val1, val2)
#define DEBUG_CHECK_NE(val1, val2) CHECK_NE(val1, val2)
#define DEBUG_CHECK_LE(val1, val2) CHECK_LE(val1, val2)
#define DEBUG_CHECK_LT(val1, val2) CHECK_LT(val1, val2)
#define DEBUG_CHECK_GE(val1, val2) CHECK_GE(val1, val2)
#define DEBUG_CHECK_GT(val1, val2) CHECK_GT(val1, val2)
#define DEBUG_CHECK_DOUBLE_EQ(val1, val2) CHECK_DOUBLE_EQ(val1, val2)
#define DEBUG_CHECK_NEAR(val1, val2, margin) CHECK_NEAR(val1, val2, margin)
