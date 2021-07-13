/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *  Copyright (c) 2016, ETH Zurich, Wyss Zurich, Zurich Eye
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * 
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Aug 30, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Zurich Eye
 *********************************************************************************/

/**
 * @file PoseParameterBlock.cpp
 * @brief Source file for the PoseParameterBlock class.
 * @author Stefan Leutenegger
 */

#include "svo/ceres_backend/pose_parameter_block.hpp"

namespace svo {
namespace ceres_backend {

// Default constructor (assumes not fixed).
PoseParameterBlock::PoseParameterBlock()
    : ParameterBlock::ParameterBlock()
{
  setFixed(false);
}

// Trivial destructor.
PoseParameterBlock::~PoseParameterBlock() {}

// Constructor with estimate.
PoseParameterBlock::PoseParameterBlock(const Transformation& T_WS, uint64_t id)
{
  setEstimate(T_WS);
  setId(id);
  setFixed(false);
}

// setters
// Set estimate of this parameter block.
void PoseParameterBlock::setEstimate(const Transformation& T_WS)
{
  const Eigen::Vector3d& r = T_WS.getPosition();
  const Quaternion& q = T_WS.getRotation();
  parameters_[0] = r[0];
  parameters_[1] = r[1];
  parameters_[2] = r[2];
  parameters_[3] = q.x();  // x
  parameters_[4] = q.y();  // y
  parameters_[5] = q.z();  // z
  parameters_[6] = q.w();  // w
}

// getters
// Get estimate.
Transformation PoseParameterBlock::estimate() const
{
  return Transformation(
      Eigen::Vector3d(parameters_[0], parameters_[1], parameters_[2]),
      Eigen::Quaterniond(parameters_[6], parameters_[3], parameters_[4],
                         parameters_[5]));
}

}  // namespace ceres_backend
}  // namespace svo
