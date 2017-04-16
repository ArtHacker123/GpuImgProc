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
    void process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<cl_int2>& coords, float threshold, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& event, std::vector<cl::Event>* pWaitEvent = 0);
    void process_cartesian(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<cl_int2>& coords, float threshold, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& event, std::vector<cl::Event>* pWaitEvent = 0);
    void process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::OptFlowData>& flowData, float threshold, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& event, std::vector<cl::Event>* pWaitEvent = 0);
    void process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::HoughData>& houghData, size_t threshold, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& event, std::vector<cl::Event>* pWaitEvent = 0);

private:
    void init(size_t warpSize);
    void adjustEventSize(size_t count, std::vector<cl::Event>& event);
    void createIntBuffer(size_t buffSize, std::vector<cl::Event>& event);
    void doScan(const cl::CommandQueue& queue, std::vector<cl::Event>& event);
    void workGroupMultipleAdjust(const cl::CommandQueue& queue);

private:
    size_t mWgrpSize;
    const size_t mScanBlkSize;
    const size_t mReduceBlkSize;

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

    Ocl::DataBuffer<int> mBuffTemp;
    std::unique_ptr< Ocl::DataBuffer<int> > mBuffReduce;

    std::vector< std::vector<cl::Event> > mWaitList;

    static const std::string sSource;
};

};
