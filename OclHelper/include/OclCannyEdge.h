#pragma once

#include "OclDataTypes.h"
#include "OclDataBuffer.h"
#include <memory>

namespace Ocl
{

class CannyEdgePrv;

class CannyEdge
{
public:
    CannyEdge(const cl::Context& ctxt);
    ~CannyEdge();

    void process(const cl::CommandQueue& queue, const cl::Image2D& inImage, cl::Image2D& outImage, float minThresh, float maxThresh,
                 std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent = 0);
    void process(const cl::CommandQueue& queue, const cl::ImageGL& inImage, cl::ImageGL& outImage, float minThresh, float maxThresh,
                 std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent = 0);

private:
    std::unique_ptr<Ocl::CannyEdgePrv> mPrv;
};

};

