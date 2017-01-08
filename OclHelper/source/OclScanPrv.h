#pragma once

#include "OclDataBuffer.h"

namespace Ocl
{

class ScanPrv
{
public:
    ScanPrv(const cl::Context& ctxt);
    ~ScanPrv();

public:
    size_t process(const cl::CommandQueue& queue, Ocl::DataBuffer<int>& buffer);

private:
    void init(size_t warpSize);
    void workGroupMultipleAdjust(const cl::CommandQueue& queue);

private:
    size_t mWgrpSize;
    const size_t mDepth;
    const size_t mBlkSize;
    const cl::Context& mContext;

    cl::Buffer mIntBuff;

    cl::Program mProgram;

    cl::Kernel mScanKernel;
    cl::Kernel mAddResKernel;
    cl::Kernel mGatherScanKernel;

    static const char sSource[];
};

};
