/*
 * Contains functions to extract features from an image and a directory with
 * images,
 * create a bow vocabulary, extract bow vectors from images and compare bow
 * vectors.
 * This is based on DBoW2 library.
 */

/*
 * File:   bow.h
 * Author: kunal71091
 *
 * Created on Nov 8, 2017
 */
#pragma once

#include <vector>
#include <string>
#include <dirent.h>

// DBoW2 (courtesy: Dorian Galvez)
#include <DBoW2/DBoW2.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

// logging
#include <glog/logging.h>

#include "keyframe.h"

namespace svo
{
/*extractFeaturesFromFolder: extracts ORB features from all the images contained
 * in a folder. This is usefule for vocabulary creation*/
void extractFeaturesFromFolder(const std::string& path_to_folder,
                               std::vector<std::vector<cv::Mat> >* features);

/*extractFeaturesFromImage: extracts ORB features from a given image*/
void extractBoWFeaturesFromImage(const cv::Mat& image,
                                 std::vector<cv::Point2f>* keypoints_pt2f,
                                 std::vector<cv::Mat>* feature);

/*createVoc: creates an OrbVocabulary based on specifications and saves it in
 * save_path*/
OrbVocabulary createVoc(const std::vector<std::vector<cv::Mat> >& features,
                        const std::string& save_path,
                        const std::string& voc_name, const int k, const int L,
                        const DBoW2::WeightingType& weight,
                        const DBoW2::ScoringType& score);

/*loadVoc: loads a vocabulary from path*/
OrbVocabulary loadVoc(const std::string& path_to_vocabulary);

/*createBOW: Creates a BOW vector using features from an image and a vocabulary.
 * It also stores the node ID of each feature in node_id vector*/
void createBOW(const std::vector<cv::Mat>& feature, const OrbVocabulary& voc,
               DBoW2::BowVector* v, std::vector<int>* node_id);

/*getNodeID: gets node ID vector from the feature vector using a vocabulary*/
void getNodeID(const std::vector<cv::Mat>& feature, const OrbVocabulary& voc,
               const int levelup, std::vector<int>* node_ids);

/*extractFeaturesFromSVOKeypoints: Extracts ORB features from given SVO
 * keypoints*/
void extractFeaturesFromSVOKeypoints(
    const cv::Mat& image, std::vector<cv::Point3f>* svo_landmarks,
    std::vector<int>* svo_landmark_ids,
    std::vector<int>* svo_trackIDs, std::vector<cv::Point2f>* svo_keypoints,
    BearingVecs* svo_bearingvectors, std::vector<double>* svo_depthvector,
    FeatureTypes* svo_featuretype_vector,
    std::vector<size_t>* svo_originalindicesvec,
    std::vector<cv::Mat>* svo_features, cv::Mat* svo_descriptors);

/*changeStructure: converts a cv::Mat to std::vector */
void changeStructure(const cv::Mat& plain, std::vector<cv::Mat>* out);

/*compareBOWs: compares two bag of words vectors and returns similarity score*/
double compareBOWs(const DBoW2::BowVector& v1, const DBoW2::BowVector& v2,
                   const OrbVocabulary& voc);

/*imgFilter: Filters images from other files in a folder*/
int imgFilter(const struct dirent* dir_ent);

}  // namespace svo
