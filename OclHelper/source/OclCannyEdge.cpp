#include "OclCannyEdge.h"
#include "OclCannyEdgePrv.h"

using namespace Ocl;

CannyEdge::CannyEdge(const cl::Context& ctxt)
    :mPrv(new CannyEdgePrv(ctxt))
{
}

CannyEdge::~CannyEdge()
{
}

size_t CannyEdge::process(const cl::CommandQueue& queue, const cl::Image2D& inImage, cl::Image2D& outImage, float minThresh, float maxThresh)
{
    return mPrv->process(queue, inImage, outImage, minThresh, maxThresh);
}

size_t CannyEdge::process(const cl::CommandQueue& queue, const cl::ImageGL& inImage, cl::ImageGL& outImage, float minThresh, float maxThresh)
{
    return mPrv->process(queue, inImage, outImage, minThresh, maxThresh);
}
