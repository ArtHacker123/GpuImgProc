#pragma once

#include <CL/cl.hpp>
#include <memory>

#include "OclDataBuffer.h"

namespace Ocl
{

class HistogramPrv
{
public:
    HistogramPrv(const cl::Context& ctxt);
    ~HistogramPrv();

    void compute(const cl::CommandQueue& queue, const cl::ImageGL& image, Ocl::DataBuffer<int>& rgbBins,
                 std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent);
    void compute(const cl::CommandQueue& queue, const cl::Image2D& image, Ocl::DataBuffer<int>& rgbBins,
                 std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent);

private:
    void init();
    void createTempHistBuffer(size_t size);
    void accumTempHist(const cl::CommandQueue& queue, size_t count, Ocl::DataBuffer<int>& rgbBins, std::vector<cl::Event>& events);
    void computeTempHist(const cl::CommandQueue& queue, const cl::Image& image, size_t& count, std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent);

private:
    const cl::Context& mContext;
    std::unique_ptr< Ocl::DataBuffer<int> > mTempBuff;

    cl::Program mPgm;

    cl::Kernel mAccHist;
    cl::Kernel mTempHistFloat;
    cl::Kernel mTempHistUint8;

    std::vector<cl::Event> mWaitEvent;

    static const char sSource[];
};

};
