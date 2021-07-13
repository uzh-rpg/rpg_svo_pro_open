#pragma once

#include <streambuf>
#include <ostream>

namespace svo
{
class NullBuffer : public std::streambuf
{
public:
  int overflow(int c)
  {
    return c;
  }
};

class NullStream : public std::ostream
{
public:
  NullStream() : std::ostream(&buffer_)
  {
  }
private:
  NullBuffer buffer_;
};

extern NullStream kNullOutput;
}
