/*
 * read_file.cpp
 *
 *  Created on: Nov 9, 2017
 *      Author: kunal71091
 */

/*
 * Function Definitions from readFile.h
 */

#include "svo/online_loopclosing/read_file.h"

using namespace std;

namespace svo
{
void readFeatureAndLandmarkData(
    const std::string& pathToFolder, const std::string& frameID1,
    const std::string& frameID2, std::vector<cv::Point2f>* keypoints1,
    std::vector<cv::Point2f>* keypoints2, std::vector<cv::Point3f>* landmarks1,
    std::vector<cv::Point3f>* landmarks2, std::vector<int>* trackIDs1,
    std::vector<int>* trackIDs2)
{
  stringstream feature1FileName, landmark1FileName, feature2FileName,
      landmark2FileName, trackIDs1FileName, trackIDs2FileName;
  feature1FileName << pathToFolder << "/" << frameID1 << "_features.txt";
  feature2FileName << pathToFolder << "/" << frameID2 << "_features.txt";
  landmark1FileName << pathToFolder << "/" << frameID1 << "_landmarks.txt";
  landmark2FileName << pathToFolder << "/" << frameID2 << "_landmarks.txt";
  trackIDs1FileName << pathToFolder << "/" << frameID1 << "_trackIDs.txt";
  trackIDs2FileName << pathToFolder << "/" << frameID2 << "_trackIDs.txt";

  ifstream trackIDFile1(trackIDs1FileName.str().c_str());
  if (trackIDFile1)
  {
    int tvalue1;
    while (trackIDFile1 >> tvalue1)
      trackIDs1->push_back(tvalue1);
  }
  trackIDFile1.close();

  ifstream trackIDFile2(trackIDs2FileName.str().c_str());
  if (trackIDFile2)
  {
    int tvalue2;
    while (trackIDFile2 >> tvalue2)
      trackIDs2->push_back(tvalue2);
  }
  trackIDFile2.close();

  ifstream featureFile1(feature1FileName.str().c_str());
  if (featureFile1)
  {
    cv::Point2f point1;
    static int count1;
    double value1;
    while (featureFile1 >> value1)
    {
      count1++;
      if (count1 % 2)
        point1.x = value1;
      else
      {
        point1.y = value1;
        keypoints1->push_back(point1);
      }
    }
  }
  featureFile1.close();

  ifstream featureFile2(feature2FileName.str().c_str());
  if (featureFile2)
  {
    cv::Point2f point2;
    static int count2;
    double value2;
    while (featureFile2 >> value2)
    {
      count2++;
      if (count2 % 2)
        point2.x = value2;
      else
      {
        point2.y = value2;
        keypoints2->push_back(point2);
      }
    }
  }
  featureFile2.close();

  ifstream landmarkFile1(landmark1FileName.str().c_str());
  if (landmarkFile1)
  {
    cv::Point3f lpoint1;
    static int lcount1;
    static int i;
    double lvalue1;
    while (landmarkFile1 >> lvalue1)
    {
      lcount1++;
      if (lcount1 % 3 != 0)
      {
        i++;
        if (i % 2)
          lpoint1.x = lvalue1;
        else
          lpoint1.y = lvalue1;
      }

      else
      {
        lpoint1.z = lvalue1;
        landmarks1->push_back(lpoint1);
        i = 0;
      }
    }
  }
  landmarkFile1.close();

  ifstream landmarkFile2(landmark2FileName.str().c_str());
  if (landmarkFile2)
  {
    cv::Point3f lpoint2;
    static int lcount2;
    static int j;
    double lvalue2;
    while (landmarkFile2 >> lvalue2)
    {
      lcount2++;
      if (lcount2 % 3 != 0)
      {
        j++;
        if (j % 2)
          lpoint2.x = lvalue2;
        else
          lpoint2.y = lvalue2;
      }

      else
      {
        lpoint2.z = lvalue2;
        landmarks2->push_back(lpoint2);
        j = 0;
      }
    }
  }
  landmarkFile2.close();
}

cv::Mat readCampose(const string& pathToFolder, const string& frameID, int rows,
                    int cols)
{
  double m;
  cv::Mat campose = cv::Mat(4, 4, CV_64F, double(0));  // Matrix to store values

  stringstream filename;
  filename << pathToFolder << "/" << frameID << "_campose.txt";

  ifstream fileStream(filename.str().c_str());
  int cnt = 0;  // index starts from 0
  while (fileStream >> m)
  {
    int temprow = cnt / cols;
    int tempcol = cnt % cols;
    campose.at<double>(temprow, tempcol) = m;
    cnt++;
  }
  return campose;
}

}  // namespace svo
