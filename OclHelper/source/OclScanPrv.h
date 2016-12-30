#pragma once

#include "OclDataBuffer.h"

namespace Ocl
{

class ScanPrv
{
public:
    ScanPrv(cl::Context& ctxt, cl::CommandQueue& queue);
    ~ScanPrv();

public:
    size_t process(Ocl::DataBuffer<int>& buffer);

private:
	void init(int warp_size);

private:
    const int mDepth;
    const int mBlkSize;
    cl::Context& mContext;
    cl::CommandQueue& mQueue;

    cl::Buffer mIntBuff;

    cl::Program mProgram;

    cl::Kernel mScanKernel;
    cl::Kernel mAddResKernel;
    cl::Kernel mGatherScanKernel;

    static const char sSource[];
};

};
