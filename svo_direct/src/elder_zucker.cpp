#include <svo/direct/elder_zucker.h>
#ifdef __SSSE3__
#include <tmmintrin.h>
#endif
#include <boost/math/special_functions/erf.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace svo {
namespace elder_zucker {

void detectEdges(
    const std::vector<cv::Mat>& img_pyr,
    const double sigma,
    cv::Mat& edge_map,
    cv::Mat& level_map)
{
//  printf("detect edges\n");
  const float pi = 3.14159265358979323846264;
  const float sn = sigma;
  const float alpha_p = 2e-7;
  const int n_levels = img_pyr.size()-1;

  // STEP-1: Use local scale control to realiably estimate the intensity
  // gradient at each image point.
  std::vector<cv::Mat> img_pyr_smoothed(n_levels);
  std::vector<cv::Mat> angle_pyr(n_levels);
  for(int L=0; L<n_levels; ++L)
  {
    // smooth image pyramid
    cv::GaussianBlur(img_pyr[L], img_pyr_smoothed[L], cv::Size(3,3), 0);

    // compute image first derivative
    const int delta=0;
    cv::Mat dx, dy;
    cv::Scharr(img_pyr_smoothed[L], dx, CV_16S, 1, 0, 1, delta, cv::BORDER_DEFAULT );
    cv::Scharr(img_pyr_smoothed[L], dy, CV_16S, 0, 1, 1, delta, cv::BORDER_DEFAULT );

    // compute critical threshold 1
    const float scale = L+1;
    const float s1 = sn * (1.0 / (2.0*sqrt(2.0*pi)*scale*scale));
    const float c1 = s1 * sqrt(-2.0*log(alpha_p));
//    printf("c1 = %f\n", c1);

    // compute angle and magnitude in angle direction
    const int n_rows = img_pyr_smoothed[L].rows;
    const int n_cols = img_pyr_smoothed[L].cols;
    angle_pyr[L] = cv::Mat(dx.size(), CV_32F);
    for(int y=0; y<n_rows; ++y)
    {
      int16_t* p_dx = dx.ptr<int16_t>(y);
      int16_t* p_dy = dy.ptr<int16_t>(y);
      float* p_a = angle_pyr[L].ptr<float>(y);
      for(int x=0; x<n_cols; ++x)
      {
        p_a[x] = std::atan2(p_dy[x], p_dx[x]);
        float mag = std::cos(p_a[x])*p_dx[x]+std::sin(p_a[x])*p_dy[x];
        if(std::abs(mag) < c1)
          p_a[x]=0;
      }
    }
  }

  // find minimum level
  cv::Mat angle(img_pyr_smoothed[0].size(), CV_32FC1, cv::Scalar(0));
  for(int y=0; y<img_pyr_smoothed[0].rows; ++y)
  {
    float* p_a = angle.ptr<float>(y);
    uint8_t* p_lev = level_map.ptr<uint8_t>(y);
    for(int x=0; x<img_pyr_smoothed[0].cols; ++x)
    {
      for(int L=0; L<n_levels; ++L)
      {
        const float a = angle_pyr[L].at<float>(y/(1<<L), x/(1<<L));
        if(a != 0.0) // TODO: what if angle is actually 0.0?
        {
          p_a[x] = a;
          break;
        }
      }
    }
  }

  std::vector<cv::Mat> lap_of_gau_pyr(img_pyr_smoothed.size());
  for(int L=0; L<n_levels; ++L)
  {
    // compute image second derivative
    cv::Mat dxdx1, dydy1, dxdy1;
    getCovarEntries(img_pyr_smoothed[L], dxdx1, dydy1, dxdy1);

    // smooth
    cv::Mat dxdx, dydy, dxdy;
    filterGauss3by316S(dxdx1, dxdx);
    filterGauss3by316S(dydy1, dydy);
    filterGauss3by316S(dxdy1, dxdy);

    // compute critical threshold 2
    const float scale = L+1;
    const float s2 = sn / (4.0 * sqrt(pi/3.0)*scale*scale*scale);
    const float c2 = sqrt(2.0) * s2 * (boost::math::erf_inv(1-alpha_p));

    // compute laplacian of gaussians
    const int n_rows = img_pyr_smoothed[L].rows;
    const int n_cols = img_pyr_smoothed[L].cols;
    lap_of_gau_pyr[L] = cv::Mat(img_pyr_smoothed[L].size(), CV_32F);
    for(int y=0; y<n_rows; ++y)
    {
      int16_t* p_dxdx = dxdx.ptr<int16_t>(y);
      int16_t* p_dxdy = dxdy.ptr<int16_t>(y);
      int16_t* p_dydy = dydy.ptr<int16_t>(y);
      float* p_a = angle.ptr<float>(y);
      float* p_l = lap_of_gau_pyr[L].ptr<float>(y);
      for(int x=0; x<n_cols; ++x)
      {
        const float ca = cos(p_a[x]);
        const float sa = sin(p_a[x]);
        p_l[x] = (ca*ca*p_dxdx[x])+(sa*sa*p_dydy[x])-(2*ca*sa*p_dxdy[x]);
        if(fabs(p_l[x]) < c2)
          p_l[x]=0;
      }
    }
  }

  // find minimum level
  edge_map = cv::Mat(img_pyr_smoothed[0].size(), CV_32FC1, cv::Scalar(0));
  level_map = cv::Mat(img_pyr_smoothed[0].size(), CV_8UC1, cv::Scalar(0));
  for(int y=0; y<img_pyr_smoothed[0].rows; ++y)
  {
    float* p_e = edge_map.ptr<float>(y);
    uint8_t* p_lev = level_map.ptr<uint8_t>(y);
    for(int x=0; x<img_pyr_smoothed[0].cols; ++x)
    {
      for(int L=0; L<n_levels; ++L)
      {
        const float e = lap_of_gau_pyr[L].at<float>(y/(1<<L), x/(1<<L));
        if(e != 0.0)
        {
          p_e[x] = e;
          p_lev[x] = L;
          break;
        }
      }
    }
  }
}


void getCovarEntries(
    const cv::Mat& src,
    cv::Mat& dxdx,
    cv::Mat& dydy,
    cv::Mat& dxdy)
{
#ifdef __SSSE3__
  cv::Mat kernel=cv::Mat::zeros(3,3,CV_8S);
  kernel.at<char>(0,0)=3*8;
  kernel.at<char>(1,0)=10*8;
  kernel.at<char>(2,0)=3*8;
  kernel.at<char>(0,2)=-3*8;
  kernel.at<char>(1,2)=-10*8;
  kernel.at<char>(2,2)=-3*8;

  const unsigned int X=3; // kernel size
  const unsigned int Y=3; // kernel size
  const unsigned int cx=1;
  const unsigned int cy=1;

  // dest will be 16 bit
  dxdx=cv::Mat::zeros(src.rows,src.cols,CV_16S);
  dydy=cv::Mat::zeros(src.rows,src.cols,CV_16S);
  dxdy=cv::Mat::zeros(src.rows,src.cols,CV_16S);

  const unsigned int maxJ=((src.cols-2)/16)*16;
  const unsigned int maxI=src.rows-2;
  const unsigned int stride=src.cols;

  __m128i mask_hi = _mm_set_epi8(0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,0xFF);
  __m128i mask_lo = _mm_set_epi8(0xFF,0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00);

  for(unsigned int i=0; i<maxI; ++i)
  {
    bool end=false;
    for(unsigned int j=0; j<maxJ; )
    {
      //__m128i result = _mm_set_epi16 ( -127,-127,-127,-127,-127,-127,-127,-127,-127,-127);
      __m128i result_hi_dx = _mm_set_epi16 ( 0,0,0,0,0,0,0,0);
      __m128i result_lo_dx = _mm_set_epi16 ( 0,0,0,0,0,0,0,0);
      __m128i result_hi_dy = _mm_set_epi16 ( 0,0,0,0,0,0,0,0);
      __m128i result_lo_dy = _mm_set_epi16 ( 0,0,0,0,0,0,0,0);

      // enter convolution with kernel
      for(unsigned int x=0;x<X;++x)
      {
        //if(dx&&x==1)continue; // jump, 0 kernel
        for(unsigned int y=0;y<Y;++y)
        {
          //if(!dx&&y==1)continue; // jump, 0 kernel
          const char m_dx=kernel.at<char>(y,x);
          const char m_dy=kernel.at<char>(x,y);
          __m128i mult_dx = _mm_set_epi16(m_dx,m_dx,m_dx,m_dx,m_dx,m_dx,m_dx,m_dx);
          __m128i mult_dy = _mm_set_epi16(m_dy,m_dy,m_dy,m_dy,m_dy,m_dy,m_dy,m_dy);
          uchar* p=(src.data+(stride*(i+y))+x+j);
          __m128i i0 = _mm_loadu_si128 ((__m128i*)p);
          __m128i i0_hi=_mm_and_si128(i0,mask_hi);
          __m128i i0_lo=_mm_srli_si128(_mm_and_si128(i0,mask_lo),1);

          if(m_dx!=0)
          {
            __m128i i_hi_dx = _mm_mullo_epi16 (i0_hi, mult_dx);
            __m128i i_lo_dx = _mm_mullo_epi16 (i0_lo, mult_dx);
            result_hi_dx=_mm_add_epi16(result_hi_dx,i_hi_dx);
            result_lo_dx=_mm_add_epi16(result_lo_dx,i_lo_dx);
          }

          if(m_dy!=0)
          {
            __m128i i_hi_dy = _mm_mullo_epi16 (i0_hi, mult_dy);
            __m128i i_lo_dy = _mm_mullo_epi16 (i0_lo, mult_dy);
            result_hi_dy=_mm_add_epi16(result_hi_dy,i_hi_dy);
            result_lo_dy=_mm_add_epi16(result_lo_dy,i_lo_dy);
          }
        }
      }

      // calculate covariance entries - remove precision (ends up being 4 bit), then remove 4 more bits
      __m128i i_hi_dx_dx = _mm_srai_epi16(_mm_mulhi_epi16 (result_hi_dx, result_hi_dx),4);
      __m128i i_hi_dy_dy = _mm_srai_epi16(_mm_mulhi_epi16 (result_hi_dy, result_hi_dy),4);
      __m128i i_hi_dx_dy = _mm_srai_epi16(_mm_mulhi_epi16 (result_hi_dy, result_hi_dx),4);
      __m128i i_lo_dx_dx = _mm_srai_epi16(_mm_mulhi_epi16 (result_lo_dx, result_lo_dx),4);
      __m128i i_lo_dy_dy = _mm_srai_epi16(_mm_mulhi_epi16 (result_lo_dy, result_lo_dy),4);
      __m128i i_lo_dx_dy = _mm_srai_epi16(_mm_mulhi_epi16 (result_lo_dy, result_lo_dx),4);

      // store
      uchar* p_lo_dxdx=(dxdx.data+(2*stride*(i+cy)))+2*cx+2*j;
      uchar* p_hi_dxdx=(dxdx.data+(2*stride*(i+cy)))+2*cx+2*j+16;
      _mm_storeu_si128 ((__m128i*)p_hi_dxdx,_mm_unpackhi_epi16 (i_hi_dx_dx, i_lo_dx_dx));
      _mm_storeu_si128 ((__m128i*)p_lo_dxdx,_mm_unpacklo_epi16 (i_hi_dx_dx, i_lo_dx_dx));
      uchar* p_lo_dydy=(dydy.data+(2*stride*(i+cy)))+2*cx+2*j;
      uchar* p_hi_dydy=(dydy.data+(2*stride*(i+cy)))+2*cx+2*j+16;
      _mm_storeu_si128 ((__m128i*)p_hi_dydy,_mm_unpackhi_epi16 (i_hi_dy_dy, i_lo_dy_dy));
      _mm_storeu_si128 ((__m128i*)p_lo_dydy,_mm_unpacklo_epi16 (i_hi_dy_dy, i_lo_dy_dy));
      uchar* p_lo_dxdy=(dxdy.data+(2*stride*(i+cy)))+2*cx+2*j;
      uchar* p_hi_dxdy=(dxdy.data+(2*stride*(i+cy)))+2*cx+2*j+16;
      _mm_storeu_si128 ((__m128i*)p_hi_dxdy,_mm_unpackhi_epi16 (i_hi_dx_dy, i_lo_dx_dy));
      _mm_storeu_si128 ((__m128i*)p_lo_dxdy,_mm_unpacklo_epi16 (i_hi_dx_dy, i_lo_dx_dy));

      // take care about end
      j+=16;
      if(j>=maxJ&&!end)
      {
        j=stride-2-16;
        end=true;
      }
    }
  }
#endif
}

void filterGauss3by316S(
    const cv::Mat& src,
    cv::Mat& dst)
{
#ifdef __SSSE3__
  // sanity check
  const unsigned int X=3;
  const unsigned int Y=3;
  assert(X%2!=0);
  assert(Y%2!=0);
  int cx=X/2;
  int cy=Y/2;

  // dest will be 16 bit
  dst=cv::Mat::zeros(src.rows,src.cols,CV_16S);
  const unsigned int maxJ=((src.cols-2)/8)*8;
  const unsigned int maxI=src.rows-2;
  const unsigned int stride=src.cols;

  for(unsigned int i=0; i<maxI; ++i)
  {
    bool end=false;
    for(unsigned int j=0; j<maxJ; )
    {
      // enter convolution with kernel. do the multiplication with 2/4 at the same time
      __m128i i00 = _mm_loadu_si128 ((__m128i*)&src.at<short>(i,j));
      __m128i i10 = _mm_slli_epi16(_mm_loadu_si128 ((__m128i*)&src.at<short>(i+1,j)),1);
      __m128i i20 = _mm_loadu_si128 ((__m128i*)&src.at<short>(i+2,j));
      __m128i i01 = _mm_slli_epi16(_mm_loadu_si128 ((__m128i*)&src.at<short>(i,j+1)),1);
      __m128i i11 = _mm_slli_epi16(_mm_loadu_si128 ((__m128i*)&src.at<short>(i+1,j+1)),2);
      __m128i i21 = _mm_slli_epi16(_mm_loadu_si128 ((__m128i*)&src.at<short>(i+2,j+1)),1);
      __m128i i02 = _mm_loadu_si128 ((__m128i*)&src.at<short>(i,j+2));
      __m128i i12 = _mm_slli_epi16(_mm_loadu_si128 ((__m128i*)&src.at<short>(i+1,j+2)),1);
      __m128i i22 = _mm_loadu_si128 ((__m128i*)&src.at<short>(i+2,j+2));
      __m128i result = i11;

      // add up
      result=_mm_add_epi16(result,i00);
      result=_mm_add_epi16(result,i20);
      result=_mm_add_epi16(result,i02);
      result=_mm_add_epi16(result,i22);

      result=_mm_add_epi16(result,i10);
      result=_mm_add_epi16(result,i01);
      result=_mm_add_epi16(result,i12);
      result=_mm_add_epi16(result,i21);

      // store
      //uchar* p_r=(dst.data+(2*stride*(i+cy)))+2*cx+2*j;
      _mm_storeu_si128 ((__m128i*)&dst.at<short>(i+cy,j+cx),result);

      // take care about end
      j+=8;
      if(j>=maxJ&&!end)
      {
              j=stride-2-8;
              end=true;
      }
    }
  }
#endif
}

} // namespace elder_zucker
} // namespace svo


