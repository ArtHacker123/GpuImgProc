#pragma once

#include "OclScan.h"
#include "OclDataTypes.h"

#include <memory>

namespace Ocl
{

class OptFlowCompact
{
public:
    OptFlowCompact(cl::Context& ctxt, cl::CommandQueue& queue);
    ~OptFlowCompact();

public:
    size_t process(cl::Image& inpImage, Ocl::DataBuffer<Ocl::OptFlowData>& out, float value, size_t& outCount);

private:
    void createIntBuffer(size_t buffSize);

private:
    cl::Context& mContext;
    cl::CommandQueue& mQueue;

    const int mScanBlkSize;
    const int mReduceBlkSize;

    cl::Program mProgram;
    cl::Kernel mReduceKernel;
    cl::Kernel mCompactScanKernel;

    Ocl::DataBuffer<int> mOutSize;
    std::unique_ptr< Ocl::DataBuffer<int> > mBuffReduce;
    Ocl::Scan mScan;

    static const char sSource[];
};

};
