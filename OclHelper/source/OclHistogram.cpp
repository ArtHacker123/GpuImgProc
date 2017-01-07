#include "OclHistogram.h"
#include "OclHistogramPrv.h"

using namespace Ocl;

Histogram::Histogram(cl::Context& ctxt, cl::CommandQueue& queue)
    :mPrv(new Ocl::HistogramPrv(ctxt, queue))
{
}

Histogram::~Histogram()
{
}

size_t Histogram::compute(const cl::ImageGL& image, Ocl::DataBuffer<int>& rgbBins)
{
    return mPrv->compute(image, rgbBins);
}

size_t Histogram::compute(const cl::Image2D& image, Ocl::DataBuffer<int>& rgbBins)
{
    return mPrv->compute(image, rgbBins);
}
