#include <svo/test_utils/synthetic_dataset.h>

#include <random>

#include <glog/logging.h>

#include <svo/common/camera.h>
#include <svo/common/frame.h>
#include <svo/common/point.h>

#include <vikit/sample.h>

namespace svo {
namespace test_utils {

SyntheticDataset::SyntheticDataset(
    const std::string& dataset_dir, size_t cam_index, size_t first_frame_id,
    double sigma_img_noise)
  : dataset_dir_(dataset_dir)
  , cam_index_(cam_index)
  , first_frame_id_(first_frame_id)
  , sigma_img_noise_(sigma_img_noise)
{
  init();
}

bool SyntheticDataset::getNextFrame(
    size_t n_pyramid_levels,
    FramePtr& frame,
    cv::Mat* depthmap)
{
  if(!img_fs_.good() || img_fs_.eof()) // end or bad
    return false;

  // skip comment block
  while(img_fs_.peek() == '#')
    img_fs_.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  // load image
  size_t img_id;
  double stamp_seconds;
  std::string img_name;
  img_fs_ >> img_id >> stamp_seconds >> img_name;
  img_fs_.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  CHECK(img_id >= first_frame_id_) << "image list is misordered.";
  uint64_t stamp = stamp_seconds*1e9;
  std::string img_filename(dataset_dir_ + "/data/" + img_name);
  cv::Mat img(cv::imread(img_filename, 0));
  if(img.empty())
  {
    LOG(ERROR) << "Failed to read image: " << img_filename;
    return false;
  }

  // add image noise
  if(sigma_img_noise_ > 0.0)
  {
    // seed
    std::random_device rd;
    std::mt19937 mt(rd());
    std::normal_distribution<double> distribution(0.0, sigma_img_noise_);

    // add to image
    uint8_t* data_ptr = img.data;
    const size_t area = img.cols*img.rows;
    for(size_t i = 0; i < area; ++i, ++data_ptr)
      *data_ptr += distribution(mt);
  }

  // make frame
  frame.reset(new svo::Frame(cam_, img, stamp, n_pyramid_levels));
  frame->id_ = img_id;

  frame->set_T_cam_imu(ncam_->get_T_C_B(cam_index_));

  // load groundtruth trajectory
  if(gt_fs_.is_open())
  {
    size_t gt_id;
    double tx, ty, tz, qx, qy, qz, qw;
    if(gt_fs_ >> gt_id >> tx >> ty >> tz >> qx >> qy >> qz >> qw)
    {
      gt_fs_.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      if(gt_id != img_id)
      {
        LOG(ERROR) << "Groundtruth id and img id no not match."
                   << " Check your groundtruth file.";
      }
      Eigen::Quaterniond q(qw, qx, qy, qz);
      if(std::abs(q.norm()-1.0) > 0.000001)
        LOG(WARNING) << "test_utils::SyntheticDataset: Quaternion norm = " << q.norm();
      q.normalize();
      Transformation T_w_body(q, Eigen::Vector3d(tx, ty, tz));
      frame->T_f_w_ = (T_w_body * frame->T_imu_cam()).inverse();
    }
    else
      LOG(WARNING) << "Could not load groundtruth data.";
  }

  // load groundtruth depth map
  if(depth_fs_.is_open())
  {
    size_t depth_id;
    std::string depth_name;
    if(depth_fs_ >> depth_id >> depth_name)
    {
      depth_fs_.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      if(depth_id != img_id)
      {
        LOG(ERROR) << "Groundtruth id and img id no not match."
                   << " Check your groundtruth file.";
      }
      if(depthmap)
      {
        std::string depth_filename(dataset_dir_ + "/data/" + depth_name);
        vk::blender_utils::loadBlenderDepthmap(
              depth_filename.c_str(), cam_->imageWidth(), cam_->imageHeight(), *depthmap);
      }
    }
  }

  return true;
}

bool SyntheticDataset::skipNImages(size_t n)
{
  for(size_t i = 0u; i < n; ++i)
  {
    FramePtr frame;
    if(!getNextFrame(1, frame, nullptr))
      return false;
  }
  return true;
}

void SyntheticDataset::init()
{
  // load camera
  ncam_ = CameraBundle::loadFromYaml(dataset_dir_+"/calib.yaml");
  if(!ncam_)
    LOG(FATAL) << "Failed to load camera.";
  cam_ = ncam_->getCameraShared(cam_index_);

  // initializing image file list
  std::string img_filename(dataset_dir_ + "/data/images.txt");
  img_fs_.open(img_filename.c_str());
  if(!img_fs_.is_open())
    LOG(FATAL) << "Failed to load image list from '" << img_filename << "'";

  // initializing groundtruth file
  std::string gt_filename(dataset_dir_ + "/data/groundtruth.txt");
  gt_fs_.open(gt_filename.c_str());
  if(!gt_fs_.is_open())
    LOG(WARNING) << "No groundtruth trajectory found.";

  // initializing depth maps file list
  std::string depth_filename(dataset_dir_ + "/data/depthmaps.txt");
  depth_fs_.open(depth_filename.c_str());
  if(!depth_fs_.is_open())
    LOG(WARNING) << "No depthmaps trajectory found.";

  skipFrames(first_frame_id_);
}

void SyntheticDataset::skipFrames(size_t first_frame_id)
{
  while(img_fs_.good() && !img_fs_.eof())
  {
    // skip comment block
    while(img_fs_.peek() == '#')
      img_fs_.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    if(first_frame_id == 0)
      return;

    if(gt_fs_.is_open())
    {
      size_t gt_id;
      double tx, ty, tz, qx, qy, qz, qw;
      gt_fs_ >> gt_id >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
      gt_fs_.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    if(depth_fs_.is_open())
    {
      size_t depthmap_id;
      std::string depth_name;
      depth_fs_ >> depthmap_id >> depth_name;
      depth_fs_.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    size_t img_id;
    double stamp_seconds;
    std::string img_name;
    img_fs_ >> img_id >> stamp_seconds >> img_name;
    img_fs_.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    if(img_id < first_frame_id-1)
      continue;
    else
      return;
  }
  LOG(ERROR) << "Image list corrupted or first_frame_id exceeds rang listed.";
}

} // namespace test_utils
} // namespace svo
