#pragma once

#include "OclDataBuffer.h"
#include <memory>

namespace Ocl
{

class HistogramPrv;

class Histogram
{
public:
    Histogram(const cl::Context& ctxt);
    ~Histogram();

    size_t compute(const cl::CommandQueue& queue, const cl::ImageGL& image, Ocl::DataBuffer<int>& rgbBins);
    size_t compute(const cl::CommandQueue& queue, const cl::Image2D& image, Ocl::DataBuffer<int>& rgbBins);

private:
    std::unique_ptr<Ocl::HistogramPrv> mPrv;
};

};
