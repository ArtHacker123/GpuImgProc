#pragma once

#include <CL/cl.hpp>
#include <memory>

#include "OclDataBuffer.h"

namespace Ocl
{

class HistogramPrv
{
public:
    HistogramPrv(cl::Context& ctxt, cl::CommandQueue& queue);
    ~HistogramPrv();

    size_t compute(const cl::ImageGL& image, Ocl::DataBuffer<int>& rgbBins);
    size_t compute(const cl::Image2D& image, Ocl::DataBuffer<int>& rgbBins);

private:
    void createTempHistBuffer(size_t size);
    size_t computeTempHist(const cl::Image& image, size_t& count);
    size_t accumTempHist(size_t count, Ocl::DataBuffer<int>& rgbBins);

private:
    cl::Context& mContext;
    cl::CommandQueue& mQueue;
    std::unique_ptr< Ocl::DataBuffer<int> > mTempBuff;

    cl::Program mPgm;

    cl::Kernel mAccHist;
    cl::Kernel mTempHistFloat;
    cl::Kernel mTempHistUint8;

    static const char sSource[];
};

};
