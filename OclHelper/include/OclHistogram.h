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

    void compute(const cl::CommandQueue& queue, const cl::ImageGL& image, Ocl::DataBuffer<int>& rgbBins,
                 std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent = 0);
    void compute(const cl::CommandQueue& queue, const cl::Image2D& image, Ocl::DataBuffer<int>& rgbBins,
                 std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent = 0);

private:
    std::unique_ptr<Ocl::HistogramPrv> mPrv;
};

};
