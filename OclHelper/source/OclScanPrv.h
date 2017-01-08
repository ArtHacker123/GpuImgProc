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
    void init(int warp_size);
    void workGroupMultipleAdjust(const cl::CommandQueue& queue);

private:
    int mWgrpSize;
    const int mDepth;
    const int mBlkSize;
    const cl::Context& mContext;

    cl::Buffer mIntBuff;

    cl::Program mProgram;

    cl::Kernel mScanKernel;
    cl::Kernel mAddResKernel;
    cl::Kernel mGatherScanKernel;

    static const char sSource[];
};

};
