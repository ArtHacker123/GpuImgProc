#pragma once

#include "OclDataBuffer.h"
#include "OclHoughLines.h"
#include "OclCompactEdges.h"
#include "OclCompactHoughData.h"
#include <memory>

namespace Ocl
{

class HoughLinesPrv
{
public:
    HoughLinesPrv(cl::Context& ctxt, cl::CommandQueue& queue);
    ~HoughLinesPrv();

    size_t process(const cl::Image2D& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, size_t& houghCount);
    size_t process(const cl::ImageGL& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, size_t& houghCount);

private:
    void init();
    size_t computeHLT(size_t count);
    size_t nonMaxSuppress(size_t threshold);
    void createTempBuffers(const cl::Image& inpImage);

private:
    size_t mRho;
    cl::Context& mContext;
    cl::CommandQueue& mQueue;

    cl::Program mPgm;
    cl::Kernel mNmsKernel;
    cl::Kernel mHoughKernel;

    //This is very messy
    Ocl::CompactEdges mEdgeCompact;
    Ocl::CompactHoughData mHoughDataCompact;

    std::unique_ptr<cl::Image2D> mHoughImg;
    std::unique_ptr<cl::Image2D> mHoughNmsImg;
    std::unique_ptr< Ocl::DataBuffer<Ocl::Pos> > mEdgeData;

    static const char sSource[];
};

};

