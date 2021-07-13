// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <list>
#include <vector>
#include <string>
//#include <cmath>      // sin, cos
#include <memory>     // shared_ptr
#include <stdexcept>  // assert, runtime_error

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <vikit/performance_monitor.h>

#include <svo/common/logging.h>
#include <svo/common/types.h>
#include <svo/common/camera.h>
#include <svo/common/transformation.h>


namespace svo
{
  //using namespace Eigen;
  using Eigen::Vector2i;
  using Eigen::Vector2f;
  using Eigen::Vector2d;
  using Eigen::Vector3d;
  using Eigen::Matrix2f;
  using Eigen::Matrix2d;
  using Eigen::Matrix3d;
  using Eigen::Matrix;

  typedef std::shared_ptr<vk::PerformanceMonitor> PerformanceMonitorPtr;

  extern PerformanceMonitorPtr g_permon;
  #define SVO_LOG(name, value) g_permon->log(std::string(name),(value))
  #define SVO_START_TIMER(name) g_permon->startTimer((name))
  #define SVO_STOP_TIMER(name) g_permon->stopTimer((name))


  // forward declaration of modules
  class Frame;
  typedef std::shared_ptr<Frame> FramePtr;
  typedef std::weak_ptr<Frame> FrameWeakPtr;
  class FrameBundle;
  typedef std::shared_ptr<FrameBundle> FrameBundlePtr;
  struct Feature;
  typedef std::shared_ptr<Feature> FeaturePtr;
  typedef std::weak_ptr<Feature> FeatureWeakPtr;
  class Point;
  typedef std::shared_ptr<Point> PointPtr;
  class ImuHandler;
  typedef std::shared_ptr<ImuHandler> ImuHandlerPtr;
  class SparseImgAlignBase;
  typedef std::shared_ptr<SparseImgAlignBase> SparseImgAlignBasePtr;
  class Map;
  typedef std::shared_ptr<Map> MapPtr;
  class Matcher;
  typedef std::shared_ptr<Matcher> MatcherPtr;
  class SeedInverse;
  typedef SeedInverse SeedImplementation;
  typedef std::shared_ptr<SeedImplementation> SeedPtr;
  typedef std::vector<SeedPtr> Seeds;
  class AbstractDetector;
  typedef std::shared_ptr<AbstractDetector> DetectorPtr;
  typedef std::vector<cv::Mat> ImgPyr;
  typedef std::vector<FeaturePtr> Features;
  class AbstractBundleAdjustment;
  typedef std::shared_ptr<AbstractBundleAdjustment> AbstractBundleAdjustmentPtr;

  enum class BundleAdjustmentType {
    kNone,
    kGtsam,
    kCeres
  };

  struct DetectorOptions;
  struct DepthFilterOptions;
  struct ReprojectorOptions;
  struct InitializationOptions;
  struct FeatureTrackerOptions;
  struct LoopClosureOptions;
  class AbstractInitialization;
  typedef std::unique_ptr<AbstractInitialization> InitializerPtr;
  class PoseOptimizer;
  typedef std::unique_ptr<PoseOptimizer> PoseOptimizerPtr;
  class SparseImgAlign;
  typedef std::unique_ptr<SparseImgAlign> SparseImgAlignPtr;
  class DepthFilter;
  typedef std::unique_ptr<DepthFilter> DepthFilterPtr;
  class Reprojector;
  typedef std::unique_ptr<Reprojector> ReprojectorPtr;
  class LoopClosing;
  typedef std::shared_ptr<LoopClosing> LoopClosingPtr;
  class GlobalMap;
  typedef std::shared_ptr<GlobalMap> GlobalMapPtr;

} // namespace svo

