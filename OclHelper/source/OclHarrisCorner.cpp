#include "OclHarrisCorner.h"
#include "OclHarrisCornerPrv.h"

using namespace Ocl;

HarrisCorner::HarrisCorner(cl::Context& ctxt, cl::CommandQueue& queue)
    :mPrv(new HarrisCornerPrv(ctxt, queue))
{
}

HarrisCorner::~HarrisCorner()
{
}

size_t HarrisCorner::process(const cl::Image2D& inImage, DataBuffer<Ocl::Pos>& corners, float value, size_t& count)
{
    return mPrv->process(inImage, corners, value, count);
}

size_t HarrisCorner::process(const cl::ImageGL& inImage, DataBuffer<Ocl::Pos>& corners, float value, size_t& count)
{
    return mPrv->process(inImage, corners, value, count);
}
