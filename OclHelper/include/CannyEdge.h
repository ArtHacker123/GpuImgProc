#pragma once

#include "DataTypes.h"
#include "DataBuffer.h"
#include <memory>

namespace Ocl
{

class CannyEdgePrv;

class CannyEdge
{
public:
    CannyEdge(cl::Context& ctxt, cl::CommandQueue& queue);
    ~CannyEdge();

    size_t process(const cl::Image2D& inImage, cl::Image2D& outImage, float minThresh, float maxThresh);
    size_t process(const cl::ImageGL& inImage, cl::ImageGL& outImage, float minThresh, float maxThresh);

private:
    std::unique_ptr<Ocl::CannyEdgePrv> mPrv;
};

};

