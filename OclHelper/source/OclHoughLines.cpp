#include "OclHoughLines.h"
#include "OclHoughLinesPrv.h"

using namespace Ocl;

HoughLines::HoughLines(cl::Context& ctxt, cl::CommandQueue& queue)
    :mPrv(new Ocl::HoughLinesPrv(ctxt, queue))
{
}

HoughLines::~HoughLines()
{
}

size_t HoughLines::process(const cl::Image2D& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, size_t& houghCount)
{
    return mPrv->process(inpImage, minSize, hData, houghCount);
}

size_t HoughLines::process(const cl::ImageGL& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, size_t& houghCount)
{
    return mPrv->process(inpImage, minSize, hData, houghCount);
}
