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

void Histogram::compute(const cl::CommandQueue& queue, const cl::ImageGL& image, Ocl::DataBuffer<int>& rgbBins,
                          std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent)
{
    mPrv->compute(queue, image, rgbBins, events, pWaitEvent);
}

void Histogram::compute(const cl::CommandQueue& queue, const cl::Image2D& image, Ocl::DataBuffer<int>& rgbBins,
                          std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent)
{
    mPrv->compute(queue, image, rgbBins, events, pWaitEvent);
}
