// This file is part of VisionTools.
//
// Copyright 2011 Hauke Strasdat (Imperial College London)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef VIKIT_RING_BUFFER_H
#define VIKIT_RING_BUFFER_H

#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>
#include <algorithm>

namespace vk
{

template<typename T>
class RingBuffer
{
public:
  RingBuffer(int size);
  RingBuffer(int size, const T& initial_value);
  ~RingBuffer() = default;

  void push_back(const T& elem);
  bool empty() const;
  void clear();
  T get(int i);
  T getSum() const;
  T getMean() const;
  T getMedian();
  int size() const { return num_elem_; }

private:
  std::vector<T> arr_;
  int begin_ = 0;
  int end_ = -1;
  int num_elem_ = 0;
  int arr_size_;
};

template <class T>
RingBuffer<T>::RingBuffer(int size)
  : arr_(size)
  , arr_size_(size)
{}

template <class T>
RingBuffer<T>::RingBuffer(int size, const T& initial_value)
  : arr_(size, initial_value)
  , arr_size_(size)
{}

template <class T>
bool RingBuffer<T>::empty() const
{
  return (num_elem_ == 0);
}

template <class T>
void RingBuffer<T>::clear()
{
  begin_ = 0;
  end_ = -1;
  num_elem_ = 0;
}

template <class T>
void RingBuffer<T>::push_back(const T & elem)
{
  if (num_elem_<arr_size_)
  {
    end_++;
    arr_[end_] = elem;
    num_elem_++;
  }
  else{
    end_ = (end_+1)%arr_size_;
    begin_ = (begin_+1)%arr_size_;
    arr_[end_] = elem;
  }
}

template <class T>
T RingBuffer<T>::get(int i)
{
  assert(i<num_elem_);
  return arr_[(begin_+i)%arr_size_];
}

template <class T>
T RingBuffer<T>::getSum() const
{
  T sum=0;
  for(int i=0; i<num_elem_; ++i)
    sum+=arr_[i];
  return sum;
}

template <class T>
T RingBuffer<T>::getMean() const
{
  if(num_elem_ == 0)
    return 0;
  return getSum()/num_elem_;
}

template <class T>
T RingBuffer<T>::getMedian()
{
  if(num_elem_ == 0)
    return 0;
  typename std::vector<T>::iterator it = arr_.begin()+std::floor(num_elem_/2);
  std::nth_element(arr_.begin(), it, arr_.end());
  return *it;
}

} // end namespace vk

#endif // VIKIT_RING_BUFFER_H
