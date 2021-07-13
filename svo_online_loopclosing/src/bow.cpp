/*
 * bow.cpp
 *
 *  Created on: Nov 8, 2017
 *      Author: kunal71091
 */

#include "svo/online_loopclosing/bow.h"

using namespace DBoW2;

// ----------------------------------------------------------------------------

namespace svo
{
void changeStructure(const cv::Mat& plain, std::vector<cv::Mat>* out)
{
  out->resize(plain.rows);

  for (int i = 0; i < plain.rows; ++i)
  {
    (*out)[i] = plain.row(i);
  }
}

int imgFilter(const struct dirent* dir_ent)
{
  std::string fname = dir_ent->d_name;

  if (fname.find(".jpg") == std::string::npos &&
      fname.find(".png") == std::string::npos)
  {
    return 0;
  }
  return 1;
}

void extractFeaturesFromFolder(const std::string& path_to_folder,
                               std::vector<std::vector<cv::Mat> >* features)
{
  features->clear();
  features->reserve(10000);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  // Step 1: Count the number of images
  VLOG(40) << "Extracting ORB features...";
  int img_count = 0;
  DIR* dir;
  struct dirent** ent;
  if ((dir = opendir(path_to_folder.c_str())) != NULL)
  {
    // count all the images within directory
    int count = scandir(path_to_folder.c_str(), &ent, *imgFilter, alphasort);
    for (int i = 0; i < count; i++)
    {
      img_count++;
      std::stringstream ss;
      ss << path_to_folder << "/" << std::string(ent[i]->d_name);
      cv::Mat image = cv::imread(ss.str(), 0);
      std::vector<cv::Point2f> keypoints;
      std::vector<cv::Mat> feature;
      extractBoWFeaturesFromImage(image, &keypoints, &feature);
      features->push_back(feature);
    }

    closedir(dir);
  }
  else
  {
    // could not open directory
    LOG(FATAL) << "Could not open directory";
  }
}

void extractBoWFeaturesFromImage(const cv::Mat& image,
                                 std::vector<cv::Point2f>* keypoints_pt2f,
                                 std::vector<cv::Mat>* feature)
{
  feature->clear();

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cv::Mat mask;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;

  orb->detectAndCompute(image, mask, keypoints, descriptors);
  // feature.push_back(cv::Mat());
  changeStructure(descriptors, feature);
  std::vector<int> conversion_mask(keypoints.size(), 0);
  keypoints_pt2f->reserve(keypoints.size());

  for (unsigned int i = 0; i < keypoints.size(); i++)
  {
    keypoints_pt2f->push_back(keypoints[i].pt);
  }
}

OrbVocabulary createVoc(const std::vector<std::vector<cv::Mat> >& features,
                        const std::string& save_path,
                        const std::string& voc_name, const int k, const int L,
                        const WeightingType& weight, const ScoringType& score)
{
  OrbVocabulary voc(k, L, weight, score);

  VLOG(1) << "Creating a" << k << "^" << L << " vocabulary...";
  voc.create(features);
  VLOG(1) << "... done!";

  VLOG(1) << "Vocabulary information: ";
  VLOG(1) << voc;

  // save the vocabulary to disk
  VLOG(1) << "Saving vocabulary...";
  std::stringstream save_loc;
  save_loc << save_path << "/" << voc_name << ".yml.gz";
  voc.save(save_loc.str());
  VLOG(1) << "Done";

  return voc;
}

OrbVocabulary loadVoc(const std::string& path_to_vocabulary)
{
  OrbVocabulary voc(path_to_vocabulary);
  return voc;
}

void createBOW(const std::vector<cv::Mat>& feature, const OrbVocabulary& voc,
               BowVector* v, std::vector<int>* node_id)
{
  voc.transform(feature, *v);
  if (node_id == nullptr)
  {
    return;
  }
  getNodeID(feature, voc, 0, node_id);
}

double compareBOWs(const BowVector& v1, const BowVector& v2,
                   const OrbVocabulary& voc)
{
  double score = voc.score(v1, v2);

  return score;
}

void getNodeID(const std::vector<cv::Mat>& feature, const OrbVocabulary& voc,
               const int levelup, std::vector<int>* node_ids)
{
  node_ids->reserve(feature.size());
  for (unsigned int i = 0; i < feature.size(); i++)
  {
    node_ids->push_back(voc.getParentNode(voc.transform(feature[i]), levelup));
  }
}

void extractFeaturesFromSVOKeypoints(
    const cv::Mat& image, std::vector<cv::Point3f>* svo_landmarks,
    std::vector<int>* svo_landmark_ids,
    std::vector<int>* svo_trackIDs, std::vector<cv::Point2f>* svo_keypoints,
    BearingVecs* svo_bearingvectors, std::vector<double>* svo_depthvector,
    FeatureTypes* svo_featuretype_vector,
    std::vector<size_t>* svo_originalindicesvec,
    std::vector<cv::Mat>* svo_features, cv::Mat* svo_descriptors)
{
  std::vector<cv::Point2f> old_svo_keypoints;
  std::vector<cv::Point3f> old_svo_landmarks;
  std::vector<int> old_svo_landmark_ids;
  BearingVecs old_svo_bearingvectors;
  std::vector<double> old_depthvectors;
  FeatureTypes old_featuretypevector;
  std::vector<int> old_svo_trackIDs;
  std::vector<size_t> old_originalindicesvec;
  old_svo_keypoints = *svo_keypoints;
  old_svo_landmarks = *svo_landmarks;
  old_svo_landmark_ids = *svo_landmark_ids;
  old_svo_trackIDs = *svo_trackIDs;
  old_svo_bearingvectors = *svo_bearingvectors;
  old_depthvectors = *svo_depthvector;
  old_featuretypevector = *svo_featuretype_vector;
  old_originalindicesvec = *svo_originalindicesvec;
  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  std::vector<cv::KeyPoint> keypoints;
  for (size_t i = 0; i < svo_keypoints->size(); i++)
  {
    keypoints.push_back(cv::KeyPoint((*svo_keypoints)[i], 1.f));
  }
  svo_keypoints->clear();
  svo_landmarks->clear();
  svo_landmark_ids->clear();
  svo_trackIDs->clear();
  svo_bearingvectors->clear();
  svo_depthvector->clear();
  svo_featuretype_vector->clear();
  svo_originalindicesvec->clear();
  //  cv::Mat svo_descriptors;
  orb->compute(image, keypoints, *svo_descriptors);
  changeStructure(*svo_descriptors, svo_features);

  // Now we remove the landmarks and trackIDs corresponding to keypoints
  // for which features could not be extracted.
  for (unsigned int i = 0; i < keypoints.size(); i++)
  {
    svo_keypoints->push_back(keypoints[i].pt);
    int pos = std::find(old_svo_keypoints.begin() + i, old_svo_keypoints.end(),
                        svo_keypoints->back()) -
              old_svo_keypoints.begin();
    svo_landmarks->push_back(old_svo_landmarks[pos]);
    svo_landmark_ids->push_back(old_svo_landmark_ids[pos]);
    svo_trackIDs->push_back(old_svo_trackIDs[pos]);
    svo_bearingvectors->push_back(old_svo_bearingvectors[pos]);
    svo_depthvector->push_back(old_depthvectors[pos]);
    svo_featuretype_vector->push_back(old_featuretypevector[pos]);
    svo_originalindicesvec->push_back(old_originalindicesvec[pos]);
  }
}

}  // namespace svo
