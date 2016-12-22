#pragma once

#include <CL/cl.hpp>
#include <memory>

#include "DataBuffer.h"

namespace Ocl
{

class ReduceSum
{
public:
    ReduceSum(cl::Context& ctxt, cl::CommandQueue& queue);
    ~ReduceSum();

public:
    int process(Ocl::DataBuffer<int>& buffer);

private:
    void createIntBuffer(size_t buffSize);

private:
    const int mDepth;
    const int mBlkSize;
    cl::Context& mContext;
    cl::CommandQueue& mQueue;

    cl::Kernel mKernel;
    cl::Program mProgram;
    std::unique_ptr< Ocl::DataBuffer<int> > mBuff;

    static const char sSource[];
};

}