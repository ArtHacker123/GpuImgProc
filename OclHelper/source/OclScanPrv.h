#pragma once

#include "OclDataBuffer.h"

#include <vector>

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
    void adjustEventSize(size_t count);
    void workGroupMultipleAdjust(const cl::CommandQueue& queue);

private:
    size_t mWgrpSize;
    const size_t mDepth;
    const size_t mBlkSize;
    const cl::Context& mContext;

    Ocl::DataBuffer<cl_int> mBuffTemp;

    std::vector<cl::Event> mEvents;
    std::vector< std::vector<cl::Event> > mWaitList;

    cl::Program mProgram;

    cl::Kernel mAdd;
    cl::Kernel mScan;
    cl::Kernel mGather;

    static const char sSource[];
};

};
