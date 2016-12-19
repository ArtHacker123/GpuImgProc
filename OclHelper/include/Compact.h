#pragma once

#include "DataTypes.h"
#include "Scan.h"
#include "DataBuffer.h"

#include <memory>

namespace Ocl
{

class Compact
{
public:
    Compact(cl::Context& ctxt, cl::CommandQueue& queue);
    ~Compact();

public:
    size_t process(cl::Image& inpImage, Ocl::DataBuffer<Ocl::Pos>& out, float value, size_t& outCount);

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

    cl::Buffer mIntBuffScan;
    std::unique_ptr< DataBuffer<int> > mBuffReduce;
    Ocl::Scan mScan;

    static const char sSource[];
};

};
