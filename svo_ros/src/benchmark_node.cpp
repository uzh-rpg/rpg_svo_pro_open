#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <opencv2/opencv.hpp>
#include <vikit/params_helper.h>
#include <vikit/blender_utils.h>
#include <vikit/sample.h>
#include <svo/common/conversions.h>
#include <svo/common/frame.h>
#include <svo/common/point.h>
#include <svo/imu_handler.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/direct/feature_detection.h>
#include <svo/direct/feature_detection_utils.h>
#include <svo_ros/svo_interface.h>
#include <svo/test_utils/synthetic_dataset.h>
#include <svo/reprojector.h>
#include <svo_ros/visualizer.h>
#include <svo_ros/csv_dataset_reader.h>
#include <svo/ceres_backend_publisher.hpp>
#include <svo/ceres_backend/estimator_types.hpp>
#include <svo/abstract_bundle_adjustment.h>
#include <svo/ceres_backend_interface.hpp>
#ifdef SVO_LOOP_CLOSING
#include <svo/online_loopclosing/loop_closing.h>
#endif

DEFINE_int32(trial_idx, 0, "indicate the index of trials");
DEFINE_bool(stamp_estimate, true, "save stamped estimate, otherwise the id is "
                                  "used");
DEFINE_bool(benchmark_auto_restart, true, "Restart the pipeline in case of "
                                          "failure.");

namespace svo
{
class BenchmarkNode : public SvoInterface
{
#ifdef SVO_LOOP_CLOSING
  const std::string kSavePg = "stamped_pose_graph_estimate";
#endif
#ifdef SVO_GLOBAL_MAP
  const std::string kSaveBA = "stamped_ba_estimate";
#endif

public:
  BenchmarkNode(const PipelineType& type, const ros::NodeHandle& nh,
                const ros::NodeHandle& pnh);

  // general evaluation functions: general dataset
  void runBenchmark(const std::string& dataset_dir);
  template <typename MarkerType>
  void tracePose(const Transformation& pose, const MarkerType marker);

  void traceIdTimePose(const Transformation& pose, const size_t id,
                       const double time_sec);

  void traceStatus(const size_t id, const double time_sec);

  void traceSpeedBias(const svo::ViNodeState& state, const size_t id,
                      const double time_sec);

  bool loadNextImages(const std::string& dataset_basedir, std::ifstream& img_fs,
                      double* stamp_sec, size_t* img_id,
                      std::vector<cv::Mat>* imgs);

  void safeCreateTrace(const std::string& trace_fn, std::ofstream* fs);

  // Blender dataset: since we can ininitialize with GT, the error can directly
  // be comptued
  void runBlenderBenchmark(const std::string& dataset_dir,
                           bool depthmap_has_zbuffer = false);
  void traceGroundtruth(const Transformation& T_w_gt, const double timestamp);
  void tracePoseError(const Transformation& T_f_gt, const double timestamp);
  void traceDepthError(const FramePtr& frame, const cv::Mat& depthmap);
  void addNoiseToImage(cv::Mat& img, double sigma);

  // KITTI
  void runKittiBenchmark(const std::string& dataset_dir);
  void tracePoseKitti(const Transformation& T_w_f);
  // Array
  void runArrayBenchmark(const std::string& dataset_dir);

private:
  // common stuff
  int frame_count_ = 0;
  std::ofstream trace_est_pose_;
  std::ofstream trace_marked_est_pose_;
  std::ofstream trace_status_;
  std::ofstream trace_speed_bias_;

  // only used for blender dataset
  std::ofstream trace_gt_pose_;
  std::ofstream trace_trans_error_;
  std::ofstream trace_rot_error_;
  std::ofstream trace_depth_error_;

  // parameters
  double img_noise_sigma_;
  size_t sleep_us_;
  size_t first_frame_id_ = 0u;
  size_t last_frame_id_ = 0u;
  size_t blackout_start_id_ = 0u;
  size_t blackout_end_id_ = 0u;
  bool trace_only_kf_;

  // manually masking frames

  // save names
  const std::string kStampedTrajEst = "stamped_traj_estimate";
  const std::string kSaveTrajEst = "traj_estimate";
  const std::string kSaveStatus = "status";
  const std::string kSaveVelBias = "speed_bias_estimate";
  const std::string kExt = "txt";
};

BenchmarkNode::BenchmarkNode(const PipelineType& type,
                             const ros::NodeHandle& nh,
                             const ros::NodeHandle& pnh)
  : SvoInterface(type, nh, pnh)
{
  // create pose tracefile
  std::string save_suf = std::to_string(FLAGS_trial_idx);
  if (FLAGS_trial_idx < 0)
  {
    save_suf.clear();
  }
  safeCreateTrace(kSaveTrajEst + save_suf + "." + kExt, &trace_est_pose_);
  trace_est_pose_.precision(20);
  safeCreateTrace(kStampedTrajEst + save_suf + "." + kExt,
                  &trace_marked_est_pose_);
  trace_marked_est_pose_.precision(20);
  safeCreateTrace(kSaveStatus + save_suf + "." + kExt, &trace_status_);
  trace_status_.precision(20);

  if (svo_->getBundleAdjuster())
  {
    safeCreateTrace(kSaveVelBias + save_suf + "." + kExt, &trace_speed_bias_);
    trace_speed_bias_.precision(20);
  }

  // read necessary parameters
  img_noise_sigma_ = vk::param<double>(pnh_, "dataset_noise_sigma", 0.0);
  sleep_us_ = static_cast<size_t>(vk::param<int>(pnh_, "sleep_us", 0));

  first_frame_id_ =
      static_cast<size_t>(vk::getParam<int>("svo/dataset_first_frame", 0));
  last_frame_id_ =
      static_cast<size_t>(vk::getParam<int>("svo/dataset_last_frame", 0));

  trace_only_kf_ = vk::getParam<bool>("svo/trace_only_keyframes", false);

  blackout_start_id_ =
      static_cast<size_t>(vk::getParam<int>("svo/blackout_first_id", 0));
  blackout_end_id_ =
      static_cast<size_t>(vk::getParam<int>("svo/blackout_last_id", 0));
}

void BenchmarkNode::safeCreateTrace(const std::string& trace_fn,
                                    std::ofstream* fs)
{
  CHECK_NOTNULL(fs);
  const std::string full_fn = svo_->options_.trace_dir + "/" + trace_fn;
  fs->open(full_fn.c_str());
  if (fs->fail())
  {
    LOG(FATAL) << "Fail to create trace " << full_fn;
  }
  else
  {
    VLOG(1) << "Created trace " << full_fn;
  }
}

void BenchmarkNode::traceIdTimePose(const Transformation& pose, const size_t id,
                                    const double time_sec)
{
  const Eigen::Quaterniond& q = pose.getRotation().toImplementation();
  const Vector3d& p = pose.getPosition();
  trace_est_pose_ << id << " " << time_sec << " ";
  trace_est_pose_ << p.x() << " " << p.y() << " " << p.z() << " " << q.x()
                  << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
}

template <typename MarkerType>
void BenchmarkNode::tracePose(const Transformation& pose,
                              const MarkerType marker)
{
  const Eigen::Quaterniond& q = pose.getRotation().toImplementation();
  const Vector3d& p = pose.getPosition();
  trace_marked_est_pose_.precision(20);
  trace_marked_est_pose_ << marker << " ";
  trace_marked_est_pose_ << p.x() << " " << p.y() << " " << p.z() << " "
                         << q.x() << " " << q.y() << " " << q.z() << " "
                         << q.w() << std::endl;
}

void BenchmarkNode::traceStatus(const size_t id, const double time_sec)
{
  trace_status_ << id << " " << time_sec << " ";
  trace_status_ << svo_->stageStr() << " " << svo_->trackingQualityStr() << " "
                << svo_->UpdateResultStr() << std::endl;
}

void BenchmarkNode::traceSpeedBias(const svo::ViNodeState& state,
                                   const size_t id, const double time_sec)
{
  trace_speed_bias_ << id << " " << time_sec << " ";
  trace_speed_bias_ << state.get_W_v_B().x() << " " << state.get_W_v_B().y()
                    << " " << state.get_W_v_B().z() << " "
                    << state.getGyroBias().x() << " " << state.getGyroBias().y()
                    << " " << state.getGyroBias().z() << " "
                    << state.getAccBias().x() << " " << state.getAccBias().y()
                    << " " << state.getAccBias().z() << " " << std::endl;
}

bool BenchmarkNode::loadNextImages(const std::string& dataset_basedir,
                                   std::ifstream& img_fs, double* stamp_sec,
                                   size_t* img_id, std::vector<cv::Mat>* images)
{
  CHECK_NOTNULL(images);
  CHECK_NOTNULL(stamp_sec);
  CHECK_NOTNULL(img_id);
  images->clear();

  if (img_fs.peek() == '#' || img_fs.peek() == '\n')  // skip and empty line
  {
    img_fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  // load image
  img_fs >> *img_id >> *stamp_sec;
  if ((*img_id < first_frame_id_) ||
      (last_frame_id_ > 0 && *img_id >= last_frame_id_) || img_fs.eof())
  {
    img_fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    return false;
  }

  size_t n_img = svo_->cams_->numCameras();
  for (size_t i = 0; i < n_img; i++)
  {
    std::string img_name;
    img_fs >> img_name;
    std::string img_filename(dataset_basedir + "/data/" + img_name);
    cv::Mat img(cv::imread(img_filename, 0));
    if (img.empty())
    {
      LOG(ERROR) << "Reading image " << img_filename << " failed. ";
    }

    if (*img_id > blackout_start_id_ && *img_id < blackout_end_id_)
    {
      img.setTo(cv::Scalar(0));
    }

    images->push_back(img);
  }
  img_fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  return true;
}

void BenchmarkNode::runBenchmark(const std::string& dataset_dir)
{
  SVO_INFO_STREAM("Run Benchmark");

  // Load imu messages.
  if (imu_handler_)
  {
    std::string imu_filename(dataset_dir + "/data/imu.txt");
    if (!imu_handler_->loadImuMeasurementsFromFile(imu_filename))
    {
      return;
    }
  }

  // Load images.
  std::string img_filename(dataset_dir + "/data/images.txt");
  std::ifstream img_fs(img_filename.c_str());
  if (!img_fs.is_open())
  {
    SVO_ERROR_STREAM("Could not open images file " << img_filename);
    return;
  }

  while (img_fs.good() && !img_fs.eof() && ros::ok())
  {
    // load image
    size_t img_id;
    double stamp_seconds;
    std::vector<cv::Mat> images;
    if (!loadNextImages(dataset_dir, img_fs, &stamp_seconds, &img_id, &images))
    {
      continue;
    }
    int64_t stamp_nanoseconds = static_cast<int64_t>(stamp_seconds * 1e9);

    // add image to VO
    setImuPrior(stamp_nanoseconds);
    processImageBundle(images, stamp_nanoseconds);
    publishResults(images, stamp_nanoseconds);

    if (svo_->stage() == Stage::kTracking || svo_->getBundleAdjuster())
    {
      const FrameBundle& nframe = *svo_->getLastFrames();
      Transformation Twb = nframe.get_T_W_B();
      if (!trace_only_kf_ || nframe.isKeyframe())
      {
        // wait till the backend returns
        if (svo_->getBundleAdjuster())
        {
          const AbstractBundleAdjustmentPtr& ba_ptr = svo_->getBundleAdjuster();
          if (ba_ptr->getType() == BundleAdjustmentType::kCeres)
          {
            while (ba_ptr->lastOptimizedBundleId() < nframe.getBundleId())
            {
              continue;
            }
          }
          else
          {
            LOG(FATAL) << "Benchmarking not implemented yet for this backend "
                          "type.";
          }
          svo::ViNodeState vi_state;
          ba_ptr->getLastState(&vi_state);
          Twb = vi_state.get_T_W_B();
          traceSpeedBias(vi_state, img_id, stamp_seconds);
          traceIdTimePose(Twb, img_id, stamp_seconds);
        }
        // only trace pose when the state is initialized
        if (svo_->isBackendScaleInitialised() &&
            svo_->map()->numKeyframes() == svo_->options_.max_n_kfs &&
            (!svo_->hasGlobalMap() || svo_->doesGlobalMapHaveInitialBA()))
        {
          if (FLAGS_stamp_estimate)
          {
            tracePose(Twb, stamp_seconds);
          }
          else
          {
            tracePose(Twb, img_id);
          }
        }
      }
    }
    traceStatus(img_id, stamp_seconds);

    // restart if it fails!
    if (svo_->stage() == Stage::kPaused && FLAGS_benchmark_auto_restart)
    {
      svo_->start();
    }

    usleep(sleep_us_);
  }

  std::string suffix("");
  if (FLAGS_trial_idx >= 0)
  {
    suffix = std::to_string(FLAGS_trial_idx);
  }

  // finishing up
  std::vector<FramePtr> remaining_kfs;
  remaining_kfs.clear();
  svo_->map()->getSortedKeyframes(remaining_kfs);
#ifdef SVO_GLOBAL_MAP
  //  std::cout << "Benchmarking: adding the rest frames..." << std::endl;
  //  if (svo_->global_map_)
  //  {
  //    for (const FramePtr& f : remaining_kfs)
  //    {
  //      svo_->global_map_->addKeyframe(f);
  //    }
  //  }
  if (svo_->global_map_)
  {
    svo_->global_map_->finishAndQuit();
    std::cout << "... all optimization done." << std::endl;
    std::string trace_ba_name =
        svo_->options_.trace_dir + "/" + kSaveBA + suffix + "." + kExt;
    std::string trace_extrins_name =
        svo_->options_.trace_dir + "/gm_extrins" + suffix + "." + kExt;
    CHECK(svo_->global_map_->traceStampedPoses(trace_ba_name));
    CHECK(svo_->global_map_->traceExtrinsics(trace_extrins_name));
  }
#endif

#ifdef SVO_LOOP_CLOSING
  /// Trace loop closing results
  if (svo_->lc_)
  {
    vk::Timer timer;
    timer.start();
    while (!svo_->lc_->lastFinished())
    {
      timer.stop();
      if (timer.getAccumulated() > 5.0)
      {
        break;
      }
      timer.resume();
    }
    std::string trace_pg_name;
    trace_pg_name =
        svo_->options_.trace_dir + "/" + kSavePg + suffix + "." + kExt;
    std::string lc_timing_path, pg_timing_path, numquery_path;
    lc_timing_path =
        svo_->options_.trace_dir + "/" + "lc_timing" + suffix + "." + kExt;
    pg_timing_path =
        svo_->options_.trace_dir + "/" + "pg_timing" + suffix + "." + kExt;
    numquery_path =
        svo_->options_.trace_dir + "/" + "num_queries" + suffix + "." + kExt;
    if (svo_->lc_->pgo_)
    {
      svo_->lc_->tracePoseGraph(trace_pg_name);
      svo_->lc_->pgo_->traceTimingData(pg_timing_path);
    }
    svo_->lc_->traceTimingData(lc_timing_path);
    svo_->lc_->traceNumQueryData(numquery_path);
    svo_->lc_->traceClosedLoops(svo_->options_.trace_dir, suffix);
  }
#endif

#ifdef SVO_GLOBAL_MAP

#endif
  VLOG(1) << "Exit benchmark.";
}

void BenchmarkNode::traceGroundtruth(const Transformation& T_w_gt,
                                     const double timestamp)
{
  const Eigen::Quaterniond& q = T_w_gt.getRotation().toImplementation();
  const Vector3d& p = T_w_gt.getPosition();
  trace_gt_pose_.precision(15);
  trace_gt_pose_.setf(std::ios::fixed, std::ios::floatfield);
  trace_gt_pose_ << timestamp << " ";
  trace_gt_pose_.precision(6);
  trace_gt_pose_ << p.x() << " " << p.y() << " " << p.z() << " " << q.x() << " "
                 << q.y() << " " << q.z() << " " << q.w() << std::endl;
}

void BenchmarkNode::tracePoseError(const Transformation& T_f_gt,
                                   const double timestamp)
{
  const Vector3d& et = T_f_gt.getPosition();  // translation error
  trace_trans_error_.precision(15);
  trace_trans_error_.setf(std::ios::fixed, std::ios::floatfield);
  trace_trans_error_ << timestamp << " ";
  trace_trans_error_.precision(6);
  trace_trans_error_ << et.x() << " " << et.y() << " " << et.z() << " "
                     << std::endl;
  Vector3d er(vk::dcm2rpy(T_f_gt.getRotationMatrix()));  // rotation error in
                                                         // roll-pitch-yaw
  trace_rot_error_.precision(15);
  trace_rot_error_.setf(std::ios::fixed, std::ios::floatfield);
  trace_rot_error_ << timestamp << " ";
  trace_rot_error_.precision(6);
  trace_rot_error_ << er.x() << " " << er.y() << " " << er.z() << " "
                   << std::endl;
}

void BenchmarkNode::tracePoseKitti(const Transformation& T_w_f)
{
  const Eigen::Matrix4d& T = T_w_f.getTransformationMatrix();
  trace_est_pose_ << T(0, 0) << " " << T(0, 1) << " " << T(0, 2) << " "
                  << T(0, 3) << " " << T(1, 0) << " " << T(1, 1) << " "
                  << T(1, 2) << " " << T(1, 3) << " " << T(2, 0) << " "
                  << T(2, 1) << " " << T(2, 2) << " " << T(2, 3) << " "
                  << std::endl;
}

void BenchmarkNode::traceDepthError(const FramePtr& frame,
                                    const cv::Mat& depthmap)
{
  trace_depth_error_.precision(6);
  for (size_t i = 0; i < frame->numFeatures(); ++i)
  {
    if (frame->landmark_vec_[i])
    {
      const double depth_estimated = (frame->pos() - frame->pos()).norm();
      const double depth_true = depthmap.at<float>((int)frame->px_vec_(1, i),
                                                   (int)frame->px_vec_(0, i));
      trace_depth_error_ << frame->id() << " " << depth_estimated - depth_true
                         << std::endl;
    }
  }
}

void BenchmarkNode::addNoiseToImage(cv::Mat& img, double sigma)
{
  uint8_t* p = (uint8_t*)img.data;
  uint8_t* p_end = img.ptr<uint8_t>(img.rows, img.cols);
  while (p != p_end)
  {
    int val = *p + vk::Sample::gaussian(sigma) + 0.5;
    *p = std::max(std::min(val, 255), 0);
    ++p;
  }
}

void BenchmarkNode::runBlenderBenchmark(const std::string& dataset_dir,
                                        bool depthmap_has_zbuffer)
{
  SVO_INFO_STREAM("Run Blender Benchmark");

  // since initialization is done using the groundtruth for synthetic data,
  // tell the reprojector not to delete points with less than two observations
  svo_->reprojectors_.at(0)->options_.remove_unconstrained_points = false;

  // create tracefiles
  trace_gt_pose_.open(svo_->options_.trace_dir + "/groundtruth_matched.txt");
  trace_trans_error_.open(svo_->options_.trace_dir + "/translation_error.txt");
  trace_rot_error_.open(svo_->options_.trace_dir + "/orientation_error.txt");
  trace_depth_error_.open(svo_->options_.trace_dir + "/depth_error.txt");
  if (trace_trans_error_.fail() || trace_rot_error_.fail() ||
      trace_depth_error_.fail() || trace_gt_pose_.fail())
    throw std::runtime_error("Could not create tracefile. Does folder exist?");

  // process dataset
  test_utils::SyntheticDataset dataset(dataset_dir, 0, 0);

  // set first frame
  const size_t n_pyr_levels = 5;
  FramePtr ref_frame;
  cv::Mat depthmap;
  Transformation T_w_gt;
  if (dataset.getNextFrame(n_pyr_levels, ref_frame, &depthmap))
  {
    // extract features, generate features with 3D points
    DetectorOptions fast_options;
    AbstractDetector::Ptr detector =
        feature_detection_utils::makeDetector(fast_options, ref_frame->cam());
    detector->detect(ref_frame);

    if (depthmap_has_zbuffer)
    {
      SVO_INFO_STREAM("Depthmap contains z-buffer values");
    }
    else
    {
      SVO_INFO_STREAM("Depthmap contains distance to camera values");
    }

    for (size_t i = 0; i < ref_frame->num_features_; ++i)
    {
      const float depth = depthmap.at<float>(ref_frame->px_vec_(1, i),
                                             ref_frame->px_vec_(0, i));
      Eigen::Vector3d landmark_ref;
      if (depthmap_has_zbuffer)
      {
        landmark_ref =
            ref_frame->f_vec_.col(i) / ref_frame->f_vec_(2, i) * depth;
      }
      else
      {
        SVO_DEBUG_STREAM("Depth (" << ref_frame->px_vec_(0, i) << " , "
                                   << ref_frame->px_vec_(1, i)
                                   << ") = " << depth);
        landmark_ref = depth * (ref_frame->f_vec_.col(i)).normalized();
      }
      Eigen::Vector3d landmark_world =
          ref_frame->T_f_w_.inverse() * landmark_ref;
      svo::PointPtr point(new svo::Point(landmark_world));
      point->addObservation(ref_frame, i);
      ref_frame->landmark_vec_[i] = point;
    }
    SVO_INFO_STREAM("Added " << ref_frame->num_features_ << " 3d pts to the "
                                                            "reference frame.");

    svo_->setFirstFrames({ ref_frame });
    SVO_INFO_STREAM("Set reference frame.");
  }
  else
  {
    SVO_ERROR_STREAM("Could not load first frame");
    return;
  }

  // process next frames
  frame_count_ = 1;
  FramePtr cur_frame;
  while (dataset.getNextFrame(n_pyr_levels, cur_frame, nullptr) && ros::ok())
  {
    T_w_gt = cur_frame->T_f_w_.inverse();

    ++frame_count_;
    SVO_DEBUG_STREAM("Processing image " << frame_count_ << ".");

    processImageBundle({ cur_frame->img() }, cur_frame->id());
    publishResults({ cur_frame->img() }, ros::Time::now().toNSec());

    if (svo_->stage() != Stage::kTracking)
    {
      SVO_ERROR_STREAM("SVO failed before entire dataset could be processed.");
      break;
    }

    if (frame_count_ == 50)
      svo_->reprojectors_.at(0)->options_.remove_unconstrained_points = true;

    // Compute pose error and trace to file
    if (svo_->getLastFrames() && svo_->isBackendScaleInitialised())
    {
      Transformation T_f_gt(svo_->getLastFrames()->at(0)->T_f_w_ * T_w_gt);
      tracePoseError(T_f_gt, cur_frame->getTimestampSec());
      traceIdTimePose(svo_->getLastFrames()->at(0)->T_f_w_.inverse(),
                      static_cast<size_t>(cur_frame->id()),
                      cur_frame->getTimestampSec());
      // traceDepthError(vo_->lastFrame(), depthmap);
      traceGroundtruth(T_w_gt, cur_frame->getTimestampSec());
    }

    usleep(sleep_us_);
  }
}

void BenchmarkNode::runKittiBenchmark(const std::string& dataset_dir)
{
  VLOG(1) << "Run Kitti Benchmark";
  std::string dataset(dataset_dir + "/data/images.txt");
  std::ifstream dataset_fs(dataset.c_str());
  if (!dataset_fs.is_open())
  {
    std::cout << "Could not open images file: " << dataset << std::endl;
    return;
  }
  std::cout << "cam0 label = " << svo_->cams_->getCamera(0).getLabel()
            << std::endl;
  std::cout << "cam1 label = " << svo_->cams_->getCamera(1).getLabel()
            << std::endl;

  // process dataset
  while (dataset_fs.good() && !dataset_fs.eof() && ros::ok())
  {
    // skip comments
    if (dataset_fs.peek() == '#')
      dataset_fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // load data
    uint64_t id;
    double stamp;
    std::string img_l_name, img_r_name;
    dataset_fs >> id >> stamp >> img_l_name >> img_r_name;
    cv::Mat img_l_8uC1(cv::imread(dataset_dir + "/data/" + img_l_name, 0));
    cv::Mat img_r_8uC1(cv::imread(dataset_dir + "/data/" + img_r_name, 0));
    if (img_l_8uC1.empty() || img_r_8uC1.empty())
    {
      SVO_ERROR_STREAM("Reading image failed: " << img_l_name);
      return;
    }
    processImageBundle({ img_l_8uC1, img_r_8uC1 }, id);
    publishResults({ img_l_8uC1, img_r_8uC1 }, ros::Time::now().toNSec());
    if (svo_->getLastFrames())
      tracePoseKitti(svo_->getLastFrames()->at(0)->T_world_cam());
  }
  VLOG(1) << "Dataset processing finished. shutting down...";
}

void BenchmarkNode::runArrayBenchmark(const std::string& dataset_dir)
{
  // load images
  std::string img_filename(dataset_dir + "/data/images.txt");
  std::ifstream img_fs(img_filename.c_str());
  if (!img_fs.is_open())
  {
    SVO_ERROR_STREAM("Could not open images file " << img_filename);
    return;
  }

  size_t first_frame_id = vk::getParam<int>("svo/dataset_first_frame", 0);
  while (img_fs.good() && !img_fs.eof() && ros::ok())
  {
    if (img_fs.peek() == '#')  // skip comments
      img_fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // load image
    uint64_t stamp = 0;
    std::string imgname1, imgname2, imgname3, imgname4;
    img_fs >> stamp >> imgname1 >> imgname2 >> imgname3 >> imgname4;
    if (stamp < first_frame_id)
      continue;
    std::string();
    cv::Mat img1(cv::imread(dataset_dir + "/data/" + imgname1, 0));
    cv::Mat img2(cv::imread(dataset_dir + "/data/" + imgname2, 0));
    cv::Mat img3(cv::imread(dataset_dir + "/data/" + imgname3, 0));
    cv::Mat img4(cv::imread(dataset_dir + "/data/" + imgname4, 0));
    if (img1.empty() || img2.empty() || img3.empty() || img4.empty())
    {
      SVO_ERROR_STREAM("Reading image " << dataset_dir << "/data/" << imgname1
                                        << " failed.");
      return;
    }

    // add image to VO
    processImageBundle({ img1, img2, img3, img4 }, stamp);
    publishResults({ img1, img2, img3, img4 }, stamp);

    cv::waitKey(10);
  }
}

}  // namespace svo

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InstallFailureSignalHandler();

  ros::init(argc, argv, "svo");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  std::string benchmark_dir(
      vk::param<std::string>(pnh, "dataset_directory", "/tmp"));
  if (vk::param<bool>(pnh, "dataset_is_kitti", false))
  {
    svo::BenchmarkNode benchmark(svo::PipelineType::kStereo, nh, pnh);
    benchmark.runKittiBenchmark(benchmark_dir);
  }
  else if (vk::param<bool>(pnh, "dataset_is_blender", false))
  {
    // does depthmap contains z values or distance to the camera (default)
    const bool depthmap_has_zbuffer =
        vk::param<bool>(pnh, "depthmap_has_zbuffer", false);
    svo::BenchmarkNode benchmark(svo::PipelineType::kMono, nh, pnh);
    benchmark.runBlenderBenchmark(benchmark_dir, depthmap_has_zbuffer);
  }
  else
  {
    if (vk::param<bool>(pnh, "dataset_is_stereo", true))
    {
      svo::BenchmarkNode benchmark(svo::PipelineType::kStereo, nh, pnh);
      benchmark.runBenchmark(benchmark_dir);
    }
    else
    {
      svo::BenchmarkNode benchmark(svo::PipelineType::kMono, nh, pnh);
      benchmark.runBenchmark(benchmark_dir);
    }
  }

  SVO_INFO_STREAM("BenchmarkNodeWithCeres finished.");
  return 0;
}
