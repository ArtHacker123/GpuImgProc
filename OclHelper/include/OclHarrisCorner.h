#pragma once

#include "OclDataTypes.h"
#include "OclDataBuffer.h"
#include <memory>

namespace Ocl
{

class HarrisCornerPrv;

class HarrisCorner
{
public:
    HarrisCorner(cl::Context& ctxt, cl::CommandQueue& queue);
    ~HarrisCorner();

    size_t process(const cl::Image2D& inImage, DataBuffer<Ocl::Pos>& corners, float value, size_t& count);
    size_t process(const cl::ImageGL& inImage, DataBuffer<Ocl::Pos>& corners, float value, size_t& count);

private:
    std::unique_ptr<Ocl::HarrisCornerPrv> mPrv;
};

};

