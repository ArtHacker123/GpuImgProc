#include "OclCannyEdge.h"
#include "OclCannyEdgePrv.h"

using namespace Ocl;

CannyEdge::CannyEdge(cl::Context& ctxt, cl::CommandQueue& queue)
    :mPrv(new CannyEdgePrv(ctxt, queue))
{
}

CannyEdge::~CannyEdge()
{
}

size_t CannyEdge::process(const cl::Image2D& inImage, cl::Image2D& outImage, float minThresh, float maxThresh)
{
    return mPrv->process(inImage, outImage, minThresh, maxThresh);
}

size_t CannyEdge::process(const cl::ImageGL& inImage, cl::ImageGL& outImage, float minThresh, float maxThresh)
{
    return mPrv->process(inImage, outImage, minThresh, maxThresh);
}
