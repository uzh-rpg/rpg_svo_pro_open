/*
 * test_patch_score.cpp
 *
 *  Created on: Dec 4, 2012
 *      Author: cforster
 */

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <svo/direct/patch_score.h>
#include <vikit/timer.h>

namespace {

const int g_halfpatch_size = 4;
const int g_patch_size = g_halfpatch_size*2;

void copyPatch(const cv::Mat& img, int x, int y, uint8_t* patch_data)
{
  cv::Mat patch(g_patch_size, g_patch_size, CV_8U, patch_data);
  img(cv::Range(y-g_halfpatch_size, y+g_halfpatch_size),
      cv::Range(x-g_halfpatch_size, x+g_halfpatch_size)).copyTo(patch);
}

void copyPatch2(cv::Mat& img, int x, int y, uint8_t* patch_data)
{
  for(int v=0; v<g_patch_size; ++v, patch_data += g_patch_size)
  {
    uint8_t* img_ptr = img.data + (y-g_halfpatch_size+v)*img.cols + (x-g_halfpatch_size);
    for(int u=0; u<g_patch_size; ++u)
      patch_data[u] = img_ptr[u];
  }
}
void testZMSSD(cv::Mat& img)
{
  int x=300, y=200;
  typedef svo::patch_score::ZMSSD<g_halfpatch_size> PatchScore;

  // create patch
  uint8_t ref_patch[g_patch_size*g_patch_size] __attribute__ ((aligned (16)));
  uint8_t cur_patch[g_patch_size*g_patch_size] __attribute__ ((aligned (16)));

  vk::Timer t;
  for(int i=0; i<1000000; ++i)
  {
    copyPatch(img, x+10, y+10, ref_patch);
    copyPatch(img, x, y, cur_patch);
  }
  printf("Copy patch cost %f\n", t.stop());

  t.start();
  for(int i=0; i<1000000; ++i)
  {
    copyPatch2(img, x+10, y+10, ref_patch);
    copyPatch2(img, x, y, cur_patch);
  }
  printf("Copy patch cost %f\n", t.stop());

  // compute patch score
  {
    t.start();
    int b=10;
    PatchScore patch_score(ref_patch);
    for(int i=0; i<1000000; ++i)
    {
      b += patch_score.computeScore(cur_patch);
    }
    printf("Compute cost cost %f, %i\n", t.stop(), b);
  }

  // compute patch score
  uint8_t* data_ptr = img.data + (y-g_halfpatch_size)*img.cols + (x-g_halfpatch_size);
  {
    t.start();
    int c=10;
    PatchScore patch_score(ref_patch);
    for(int i=0; i<1000000; ++i)
    {
      c += patch_score.computeScore(data_ptr, img.cols);
    }
    printf("Compute cost, stride %f, %i\n", t.stop(), c);
  }

  // check results
  {
    PatchScore patch_score(ref_patch);
    printf("Score = %i\n", patch_score.computeScore(cur_patch));
  }

  // check results
  {
    PatchScore patch_score(ref_patch);
    printf("Score = %i\n", patch_score.computeScore(data_ptr, img.cols));
  }

}

} // namespace


int main(int argc, char **argv)
{
  std::string img_name("test/data/scene_000.png");
  cv::Mat img(cv::imread(img_name, 0));
  assert(!img.empty());

  testZMSSD(img);

  return 0;
}
