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

    void process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<cl_int2>& coords,
                   float threshold, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& event, std::vector<cl::Event>* pWaitEvent = 0);
    void process_cartesian(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<cl_int2>& coords,
                             float threshold, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& event, std::vector<cl::Event>* pWaitEvent = 0);
    void process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::OptFlowData>& flowData,
                   float threshold, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& event, std::vector<cl::Event>* pWaitEvent = 0);
    void process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::HoughData>& houghData,
                   size_t threshold, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& event, std::vector<cl::Event>* pWaitEvent = 0);

private:
    std::unique_ptr<Ocl::CompactPrv> mPrv;
};

};
