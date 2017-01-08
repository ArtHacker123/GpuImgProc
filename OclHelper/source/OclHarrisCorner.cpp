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

size_t HarrisCorner::process(const cl::CommandQueue& queue, const cl::Image2D& inImage, DataBuffer<Ocl::Pos>& corners, float value, size_t& count)
{
    return mPrv->process(queue, inImage, corners, value, count);
}

size_t HarrisCorner::process(const cl::CommandQueue& queue, const cl::ImageGL& inImage, DataBuffer<Ocl::Pos>& corners, float value, size_t& count)
{
    return mPrv->process(queue, inImage, corners, value, count);
}
