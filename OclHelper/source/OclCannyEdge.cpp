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

void CannyEdge::process(const cl::CommandQueue& queue, const cl::Image2D& inImage, cl::Image2D& outImage, float minThresh, float maxThresh,
                        std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent)
{
    mPrv->process(queue, inImage, outImage, minThresh, maxThresh, events, pWaitEvent);
}

void CannyEdge::process(const cl::CommandQueue& queue, const cl::ImageGL& inImage, cl::ImageGL& outImage, float minThresh, float maxThresh,
                        std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent)
{
    mPrv->process(queue, inImage, outImage, minThresh, maxThresh, events, pWaitEvent);
}
