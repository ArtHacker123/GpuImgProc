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

    void process(const cl::CommandQueue& queue, const cl::Image2D& inImage, DataBuffer<cl_int2>& corners, float value, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent = 0);
    void process(const cl::CommandQueue& queue, const cl::ImageGL& inImage, DataBuffer<cl_int2>& corners, float value, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent = 0);

private:
    std::unique_ptr<Ocl::HarrisCornerPrv> mPrv;
};

};

