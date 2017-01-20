#pragma once

#include "OclDataTypes.h"
#include "OclDataBuffer.h"
#include <memory>

namespace Ocl
{

class CompactPrv;

class Compact
{
public:
    Compact(const cl::Context& ctxt);
    ~Compact();

    size_t process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<cl_int2>& coords, float threshold, size_t& count);
    size_t process_cartesian(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<cl_int2>& coords, float threshold, size_t& count);
    size_t process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::OptFlowData>& flowData, float threshold, size_t& count);
    size_t process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::HoughData>& houghData, size_t threshold, size_t& count);

private:
    std::unique_ptr<Ocl::CompactPrv> mPrv;
};

};
