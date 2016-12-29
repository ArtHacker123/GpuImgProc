#pragma once

#include "OclDataBuffer.h"

namespace Ocl
{

class Scan
{
public:
    Scan(cl::Context& ctxt, cl::CommandQueue& queue);
    ~Scan();

public:
    size_t process(Ocl::DataBuffer<int>& buffer);

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
