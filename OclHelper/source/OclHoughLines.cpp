#include "OclHoughLines.h"
#include "OclHoughLinesPrv.h"

using namespace Ocl;

HoughLines::HoughLines(const cl::Context& ctxt)
    :mPrv(new Ocl::HoughLinesPrv(ctxt))
{
}

HoughLines::~HoughLines()
{
}

size_t HoughLines::process(const cl::CommandQueue& queue, const cl::Image2D& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, size_t& houghCount)
{
    return mPrv->process(queue, inpImage, minSize, hData, houghCount);
}

size_t HoughLines::process(const cl::CommandQueue& queue, const cl::ImageGL& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, size_t& houghCount)
{
    return mPrv->process(queue, inpImage, minSize, hData, houghCount);
}
