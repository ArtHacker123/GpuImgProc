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

    size_t process(const cl::CommandQueue& queue, const cl::Image2D& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, size_t& houghCount);
    size_t process(const cl::CommandQueue& queue, const cl::ImageGL& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, size_t& houghCount);

private:
    void init();
    void createTempBuffers(const cl::Image& inpImage);
    size_t computeHLT(const cl::CommandQueue& queue, size_t count);
    size_t nonMaxSuppress(const cl::CommandQueue& queue, size_t threshold);

private:
    size_t mRho;
    const cl::Context& mContext;

    cl::Program mPgm;
    cl::Kernel mNmsKernel;
    cl::Kernel mHoughKernel;

    Ocl::Compact mCompact;

    std::unique_ptr<cl::Image2D> mHoughImg;
    std::unique_ptr<cl::Image2D> mHoughNmsImg;
    std::unique_ptr< Ocl::DataBuffer<Ocl::Pos> > mEdgeData;

    static const char sSource[];
};

};

