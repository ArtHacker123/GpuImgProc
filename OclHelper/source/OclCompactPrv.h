#pragma once

#include "OclDataTypes.h"
#include "OclScan.h"

#include <memory>

namespace Ocl
{

class CompactPrv
{
public:
    CompactPrv(cl::Context& ctxt, cl::CommandQueue& queue);
    ~CompactPrv();

public:
    size_t process(const cl::Image& inpImage, Ocl::DataBuffer<Ocl::Pos>& coords, float threshold, size_t& count);
    size_t process_cartesian(const cl::Image& inpImage, Ocl::DataBuffer<Ocl::Pos>& coords, float threshold, size_t& count);
    size_t process(const cl::Image& inpImage, Ocl::DataBuffer<Ocl::OptFlowData>& flowData, float threshold, size_t& count);
    size_t process(const cl::Image& inpImage, Ocl::DataBuffer<Ocl::HoughData>& houghData, size_t threshold, size_t& count);

private:
    void init(int warp_size);
    void createIntBuffer(size_t buffSize);

private:
    cl::Context& mContext;
    cl::CommandQueue& mQueue;

    const int mScanBlkSize;
    const int mReduceBlkSize;

    cl::Program mProgram;

    cl::Kernel mReduceFloatX;
    cl::Kernel mCompactFloatX;
    cl::Kernel mCompactCartFloatX;

    cl::Kernel mReduceFloatZ;
    cl::Kernel mCompactOptFlow;

    cl::Kernel mReduceIntX;
    cl::Kernel mCompactHoughData;

    Ocl::DataBuffer<int> mOutSize;
    std::unique_ptr< Ocl::DataBuffer<int> > mBuffReduce;

    Ocl::Scan mScan;

    static const char sSource[];
};

};
