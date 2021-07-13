#pragma once

#include <memory>
#include <ros/ros.h>
#include <svo/common/camera_fwd.h>

namespace svo {

// forward declarations
class ImuHandler;
class LoopClosing;
class GlobalMap;
class FrameHandlerMono;
class FrameHandlerStereo;
class FrameHandlerArray;
class FrameHandlerDenseMono;

namespace factory {

/// Get IMU Handler.
std::shared_ptr<ImuHandler> getImuHandler(
    const ros::NodeHandle& pnh);

#ifdef SVO_LOOP_CLOSING
/// Create loop closing module
std::shared_ptr<LoopClosing> getLoopClosingModule(
    const ros::NodeHandle& pnh,
    const CameraBundlePtr& cam=nullptr);
#endif

#ifdef SVO_GLOBAL_MAP
std::shared_ptr<GlobalMap> getGlobalMap(
    const ros::NodeHandle& pnh,
    const CameraBundlePtr& ncams = nullptr);
#endif

/// Factory for Mono-SVO.
std::shared_ptr<FrameHandlerMono> makeMono(
    const ros::NodeHandle& pnh,
    const CameraBundlePtr& cam = nullptr);

/// Factory for Stereo-SVO.
std::shared_ptr<FrameHandlerStereo> makeStereo(
    const ros::NodeHandle& pnh,
    const CameraBundlePtr& cam = nullptr);

/// Factory for Camera-Array-SVO.
std::shared_ptr<FrameHandlerArray> makeArray(
    const ros::NodeHandle& pnh,
    const CameraBundlePtr& cam = nullptr);

/// Factory for Camera-Array-SVO
std::shared_ptr<FrameHandlerDenseMono> makeDenseMono(
    const ros::NodeHandle& pnh);

} // namespace factory
} // namespace mono
