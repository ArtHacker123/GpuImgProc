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

    size_t process(const cl::CommandQueue& queue, const cl::Image2D& inImage, cl::Image2D& outImage, float minThresh, float maxThresh);
    size_t process(const cl::CommandQueue& queue, const cl::ImageGL& inImage, cl::ImageGL& outImage, float minThresh, float maxThresh);

private:
    std::unique_ptr<Ocl::CannyEdgePrv> mPrv;
};

};

