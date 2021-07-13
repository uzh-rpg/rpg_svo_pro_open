#pragma once

#include <imp/core/pixel.hpp>

namespace svo {

#define USE_SINGLE_PRECISION

#ifdef USE_SINGLE_PRECISION
typedef float FloatTypeGpu;
#else
typedef double FloatTypeGpu;
#endif

#ifdef USE_SINGLE_PRECISION
typedef imp::Pixel32fC1 FloatPixelGpu;
typedef float2 Float2TypeGpu;
typedef imp::Pixel32fC2 Float2PixelGpu;
typedef float3 Float3TypeGpu;
typedef imp::Pixel32fC3 Float3PixelGpu;
typedef std::uint32_t UIntTypeGpu;
typedef imp::Pixel32uC1 UIntPixelGpu;
typedef unsigned char BoolTypeGpu;
typedef imp::Pixel8uC1 BoolPixelGpu;
#else
// use double precision
// TODO: Define Pixel64 in imp
typedef imp::Pixel64fC1 FloatPixelGpu;
typedef double2 Float2TypeGpu;
typedef imp::Pixel64fC2 Float2PixelGpu;
typedef double3 Float3TypeGpu;
typedef imp::Pixel64fC3 Float3PixelGpu;
typedef std::uint32_t UIntTypeGpu;
typedef imp::Pixel32uC1 UIntPixelGpu;
typedef unsigned char BoolTypeGpu;
typedef imp::Pixel8uC1 BoolPixelGpu;
#endif

} // namespace svo
