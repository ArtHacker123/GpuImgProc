#pragma once

#include "OclDataTypes.h"
#include "OclScan.h"

#include <memory>

namespace Ocl
{

class CompactEdges
{
public:
    CompactEdges(cl::Context& ctxt, cl::CommandQueue& queue);
    ~CompactEdges();

public:
    size_t process(const cl::Image& inpImage, Ocl::DataBuffer<Ocl::Pos>& out, float value, size_t& outCount);

private:
    void init(int warp_size);
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
