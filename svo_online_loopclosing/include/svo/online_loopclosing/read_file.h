/*
 * read_file.h
 *
 *  Created on: Nov 9, 2017
 *      Author: kunal71091
 */

/*
 * For functions to read data from text files and store into variables.
 */
#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include "svo/online_loopclosing/bow.h"
#include "svo/online_loopclosing/geometric_verification.h"

namespace svo
{
/* Function readFeatureAndLandmarkData
 * This function takes pathToFolder as an input along with two frameIDs and
 * returns vectors
 * of 2D features and 3D Landmarks*/

void readFeatureAndLandmarkData(
    const std::string& pathToFolder, const std::string& frameID1,
    const std::string& frameID2, std::vector<cv::Point2f>* keypoints1,
    std::vector<cv::Point2f>* keypoints2, std::vector<cv::Point3f>* landmarks1,
    std::vector<cv::Point3f>* landmarks2, std::vector<int>* trackIDs1,
    std::vector<int>* trackIDs2);

/* Function readCampose
 * Reads Camera pose (R | T) from a text file*/

cv::Mat readCamPose(const std::string& pathToFolder, const std::string& frameID,
                    int rows, int cols);

}  // namespace svo
