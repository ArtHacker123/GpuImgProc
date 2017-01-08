#pragma once

#include <CL/cl.hpp>
#include <memory>

#include "OclDataBuffer.h"

namespace Ocl
{

class ReduceSum
{
public:
    ReduceSum(const cl::Context& ctxt);
    ~ReduceSum();

public:
    int process(const cl::CommandQueue& queue, Ocl::DataBuffer<int>& buffer);

private:
    void createIntBuffer(size_t buffSize);

private:
    const int mDepth;
    const int mBlkSize;
    const cl::Context& mContext;

    cl::Kernel mKernel;
    cl::Program mProgram;
    std::unique_ptr< Ocl::DataBuffer<int> > mBuff;

    static const char sSource[];
};

}