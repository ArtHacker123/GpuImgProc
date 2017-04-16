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
    void process(const cl::CommandQueue& queue, Ocl::DataBuffer<int>& buffer,
                    std::vector<cl::Event>& event, std::vector<cl::Event>* pWaitEvent = 0);

private:
    void init(size_t warpSize);
    void workGroupMultipleAdjust(const cl::CommandQueue& queue);
    void adjustEventSize(size_t count, std::vector<cl::Event>& event);

private:
    size_t mWgrpSize;
    const size_t mDepth;
    const size_t mBlkSize;
    const cl::Context& mContext;

    Ocl::DataBuffer<cl_int> mBuffTemp;
    std::vector< std::vector<cl::Event> > mWaitList;

    cl::Program mProgram;

    cl::Kernel mAdd;
    cl::Kernel mScan;
    cl::Kernel mGather;

    static const std::string sSource;
};

};
