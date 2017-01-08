#include "OclHistogram.h"
#include "OclHistogramPrv.h"

using namespace Ocl;

Histogram::Histogram(const cl::Context& ctxt)
    :mPrv(new Ocl::HistogramPrv(ctxt))
{
}

Histogram::~Histogram()
{
}

size_t Histogram::compute(const cl::CommandQueue& queue, const cl::ImageGL& image, Ocl::DataBuffer<int>& rgbBins)
{
    return mPrv->compute(queue, image, rgbBins);
}

size_t Histogram::compute(const cl::CommandQueue& queue, const cl::Image2D& image, Ocl::DataBuffer<int>& rgbBins)
{
    return mPrv->compute(queue, image, rgbBins);
}
