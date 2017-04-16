#pragma once

#include "OclDataBuffer.h"
#include "OclHoughLines.h"
#include "OclCompact.h"
#include <memory>

namespace Ocl
{

class HoughLinesPrv
{
public:
    HoughLinesPrv(const cl::Context& ctxt);
    ~HoughLinesPrv();

    void process(const cl::CommandQueue& queue, const cl::Image2D& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, Ocl::DataBuffer<cl_int>& houghCount,
                std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent);
    void process(const cl::CommandQueue& queue, const cl::ImageGL& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, Ocl::DataBuffer<cl_int>& houghCount,
                std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent);

private:
    void init();
    void createTempBuffers(const cl::Image& inpImage);
    void nonMaxSuppress(const cl::CommandQueue& queue, size_t threshold, std::vector<cl::Event>& events);
    void computeHLT(const cl::CommandQueue& queue, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& events);

private:
    size_t mRho;
    const cl::Context& mContext;
    Ocl::DataBuffer<cl_int> mEdgeCount;

    cl::Program mPgm;
    cl::Kernel mNmsKernel;
    cl::Kernel mHoughKernel;

    Ocl::Compact mCompact;

    std::vector<cl::Event> mWaitEvent[3];

    std::unique_ptr<cl::Image2D> mHoughImg;
    std::unique_ptr<cl::Image2D> mHoughNmsImg;
    std::unique_ptr< Ocl::DataBuffer<cl_int2> > mEdgeData;

    static const char sSource[];
};

};

