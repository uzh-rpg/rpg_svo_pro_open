#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <glog/logging.h>
#include <svo/common/types.h>
#include <aslam/common/macros.h>
#include <opencv2/highgui/highgui.hpp>

namespace svo {
namespace dataset {

enum class MeasurementType {
  kImu,
  kCamera,
  kFeatureTrack,
};

struct MeasurementBase
{
  ASLAM_POINTER_TYPEDEFS(MeasurementBase);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MeasurementBase() = delete;
  MeasurementBase(int64_t timestamp_nanoseconds, MeasurementType type)
    : timestamp_nanoseconds(timestamp_nanoseconds)
    , measurement_type(type)
  {}
  virtual ~MeasurementBase() = default;

  const int64_t timestamp_nanoseconds;
  const MeasurementType measurement_type;
};

struct ImuMeasurement : public MeasurementBase
{
  ASLAM_POINTER_TYPEDEFS(ImuMeasurement);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuMeasurement() = delete;
  ImuMeasurement(
      int64_t timestamp_nanoseconds,
      const Eigen::Matrix<double, 6, 1>& acc_gyro_measurement)
    : MeasurementBase(timestamp_nanoseconds, MeasurementType::kImu)
    , acc_gyro_measurement(acc_gyro_measurement)
  {}
  virtual ~ImuMeasurement() = default;

  const Eigen::Matrix<double, 6, 1> acc_gyro_measurement;
};

struct CameraMeasurement : public MeasurementBase
{
  ASLAM_POINTER_TYPEDEFS(CameraMeasurement);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CameraMeasurement() = delete;
  CameraMeasurement(
      int64_t timestamp_nanoseconds
      , size_t camera_index
      , const std::string& image_path_filename)
    : MeasurementBase(timestamp_nanoseconds, MeasurementType::kCamera)
    , camera_index(camera_index)
    , image_path_filename(image_path_filename)
  {}
  virtual ~CameraMeasurement() = default;

  cv::Mat getImage() const
  {
    cv::Mat image = cv::imread(image_path_filename, cv::IMREAD_GRAYSCALE);
    CHECK_NOTNULL(image.data);
    return image;
  }

  const size_t camera_index;
  const std::string image_path_filename;
};

struct FeatureTrackMeasurement : public MeasurementBase
{
  ASLAM_POINTER_TYPEDEFS(FeatureTrackMeasurement);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FeatureTrackMeasurement() = delete;
  FeatureTrackMeasurement(
      int64_t timestamp_nanoseconds,
      size_t camera_index,
      int track_id,
      const Eigen::Vector2d& keypoint_measurement,
      double keypoint_std_dev)
    : MeasurementBase(timestamp_nanoseconds, MeasurementType::kFeatureTrack)
    , camera_index(camera_index)
    , track_id(track_id)
    , keypoint_measurement(keypoint_measurement)
    , keypoint_std_dev(keypoint_std_dev)
  {}
  virtual ~FeatureTrackMeasurement() = default;

  const size_t camera_index;
  const int track_id;
  const Eigen::Vector2d keypoint_measurement;
  const double keypoint_std_dev;
};

} // namespace dataset

class CsvDatasetReader
{
public:
  typedef std::pair<int64_t, std::shared_ptr<dataset::MeasurementBase>> StampMeasurementPair;
  typedef std::multimap<int64_t, std::shared_ptr<dataset::MeasurementBase>> DataBuffer;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ASLAM_POINTER_TYPEDEFS(CsvDatasetReader);

  CsvDatasetReader(
      const std::string& csv_directory,
      const std::vector<size_t> imu_indices,
      const std::vector<size_t> camera_indices,
      const std::vector<size_t> track_indices);

  virtual ~CsvDatasetReader() = default;

  // Make class iterable:
  typedef DataBuffer::value_type value_type;
  typedef DataBuffer::const_iterator const_iterator;
  DataBuffer::const_iterator begin() const { return buffer_.begin(); }
  DataBuffer::const_iterator end() const { return buffer_.end(); }

private:

  void loadImuData(
      const std::string data_dir,
      const int64_t playback_delay);

  void loadCameraData(
      const std::string& data_dir,
      const size_t camera_index,
      int64_t playback_delay);

  void loadFeatureTracksData(
      const std::string& data_dir,
      const size_t camera_index,
      int64_t playback_delay);

  /// Buffer to chronologically sort the data.
  DataBuffer buffer_;
};

} // namespace svo
