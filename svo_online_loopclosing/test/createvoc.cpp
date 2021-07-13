/*
 *  createvoc.cpp
 *
 *  Use this file to create a new vocabulary. Specify the path where the images
 * have been stored in extractFeaturesFromFolder
 *  function. The Vocabulary Name and other associated parameters, have to be
 * specified.
 *
 *  Created on: Nov 8, 2017
 *  Author: kunal71091
 */

#include <fstream>
#include <ros/package.h>

#include "svo/online_loopclosing/bow.h"

using namespace std;
using namespace svo;
using namespace DBoW2;

int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    std::cerr << "Provide Image Directory Path and Vocabulary name"
              << std::endl;
    return 1;
  }
  string folder_location = std::string(argv[1]);
  // branching factor and depth levels
  const int k = 8;
  const int L = 4;
  const WeightingType weight = TF_IDF;
  const ScoringType score = BHATTACHARYYA;
  stringstream voc_save_path_;
  voc_save_path_ << ros::package::getPath("svo_online_loopclosing") << "/vocabu"
                                                                       "laries";
  string voc_save_path = voc_save_path_.str();

  string voc_name = std::string(argv[2]);
  vector<vector<cv::Mat>> features;
  extractFeaturesFromFolder(folder_location, &features);
  OrbVocabulary voc =
      createVoc(features, voc_save_path, voc_name, k, L, weight, score);

  return 0;
}
