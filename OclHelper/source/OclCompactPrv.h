#pragma once

#include "OclDataTypes.h"
#include "OclScan.h"

#include <memory>

namespace Ocl
{

class CompactPrv
{
public:
    CompactPrv(const cl::Context& ctxt);
    ~CompactPrv();

public:
    size_t process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::Pos>& coords, float threshold, size_t& count);
    size_t process_cartesian(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::Pos>& coords, float threshold, size_t& count);
    size_t process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::OptFlowData>& flowData, float threshold, size_t& count);
    size_t process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::HoughData>& houghData, size_t threshold, size_t& count);

private:
    void init(size_t warpSize);
    void createIntBuffer(size_t buffSize);
    void workGroupMultipleAdjust(const cl::CommandQueue& queue);

private:
    size_t mWgrpSize;
    const size_t mScanBlkSize;
    const size_t mReduceBlkSize;

    const cl::Context& mContext;

    cl::Program mProgram;

    cl::Kernel mReduceFloatX;
    cl::Kernel mCompactFloatX;
    cl::Kernel mCompactCartFloatX;

    cl::Kernel mReduceFloatZ;
    cl::Kernel mCompactOptFlow;

    cl::Kernel mReduceIntX;
    cl::Kernel mCompactHoughData;

    Ocl::DataBuffer<int> mOutSize;
    std::unique_ptr< Ocl::DataBuffer<int> > mBuffReduce;

    Ocl::Scan mScan;

    static const char sSource[];
};

};
