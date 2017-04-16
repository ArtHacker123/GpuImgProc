#include "OclHarrisCorner.h"
#include "OclHarrisCornerPrv.h"

using namespace Ocl;

HarrisCorner::HarrisCorner(const cl::Context& ctxt)
    :mPrv(new HarrisCornerPrv(ctxt))
{
}

HarrisCorner::~HarrisCorner()
{
}

void HarrisCorner::process(const cl::CommandQueue& queue, const cl::Image2D& inImage, DataBuffer<cl_int2>& corners, float value, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent)
{
    mPrv->process(queue, inImage, corners, value, count, events, pWaitEvent);
}

void HarrisCorner::process(const cl::CommandQueue& queue, const cl::ImageGL& inImage, DataBuffer<cl_int2>& corners, float value, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent)
{
    mPrv->process(queue, inImage, corners, value, count, events, pWaitEvent);
}
