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
    HarrisCorner(const cl::Context& ctxt);
    ~HarrisCorner();

    size_t process(const cl::CommandQueue& queue, const cl::Image2D& inImage, DataBuffer<cl_int2>& corners, float value, size_t& count);
    size_t process(const cl::CommandQueue& queue, const cl::ImageGL& inImage, DataBuffer<cl_int2>& corners, float value, size_t& count);

private:
    std::unique_ptr<Ocl::HarrisCornerPrv> mPrv;
};

};

