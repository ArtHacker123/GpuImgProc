#pragma once

#include "OclDataTypes.h"
#include "OclDataBuffer.h"

#include <memory>

namespace Ocl
{

class CompactPrv
{
public:
    CompactPrv(const cl::Context& ctxt);
    ~CompactPrv();

public:
    size_t process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<cl_int2>& coords, float threshold, size_t& count);
    size_t process_cartesian(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<cl_int2>& coords, float threshold, size_t& count);
    size_t process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::OptFlowData>& flowData, float threshold, size_t& count);
    size_t process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::HoughData>& houghData, size_t threshold, size_t& count);

private:
    void init(size_t warpSize);
    void adjustEventSize(size_t count);
    void createIntBuffer(size_t buffSize);
    void doScan(const cl::CommandQueue& queue);
    void workGroupMultipleAdjust(const cl::CommandQueue& queue);

private:
    size_t mWgrpSize;
    const size_t mScanBlkSize;
    const size_t mReduceBlkSize;

    size_t mEvtCount;
    size_t mWlistCount;

    const cl::Context& mContext;

    cl::Program mProgram;

    cl::Kernel mAdd;
    cl::Kernel mScan;
    cl::Kernel mGather;

    cl::Kernel mReduceFloatX;
    cl::Kernel mCompactFloatX;
    cl::Kernel mCompactCartFloatX;

    cl::Kernel mReduceFloatZ;
    cl::Kernel mCompactOptFlow;

    cl::Kernel mReduceIntX;
    cl::Kernel mCompactHoughData;

    Ocl::DataBuffer<int> mOutSize;
    Ocl::DataBuffer<int> mBuffTemp;
    std::unique_ptr< Ocl::DataBuffer<int> > mBuffReduce;

    std::vector<cl::Event> mEvents;
    std::vector< std::vector<cl::Event> > mWaitList;

    static const char sSource[];
};

};
