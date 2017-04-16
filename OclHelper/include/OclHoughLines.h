#pragma once

#include "OclDataTypes.h"
#include "OclDataBuffer.h"
#include <memory>

namespace Ocl
{

class HoughLinesPrv;

class HoughLines
{
public:
    HoughLines(const cl::Context& ctxt);
    ~HoughLines();

    void process(const cl::CommandQueue& queue, const cl::Image2D& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, Ocl::DataBuffer<cl_int>& houghCount,
        std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent = 0);
    void process(const cl::CommandQueue& queue, const cl::ImageGL& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, Ocl::DataBuffer<cl_int>& houghCount,
        std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent = 0);

private:
    std::unique_ptr<Ocl::HoughLinesPrv> mPrv;
};

};
