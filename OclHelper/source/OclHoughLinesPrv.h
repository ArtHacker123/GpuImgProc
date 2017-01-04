#pragma once

#include "OclDataBuffer.h"
#include "OclHoughLines.h"
#include <memory>

namespace Ocl
{

class HoughLinesPrv
{
public:
    HoughLinesPrv(cl::Context& ctxt, cl::CommandQueue& queue);
    ~HoughLinesPrv();

public:
    size_t process(const Ocl::DataBuffer<Ocl::Pos>& edgeData, size_t edgeCount, size_t width, size_t height, Ocl::HoughData& hData);

private:
    void init();
    void createTempData(const Ocl::DataBuffer<Ocl::Pos>& edgeData);

private:
    cl::Context& mContext;
    cl::CommandQueue& mQueue;

    cl::Program mPgm;
    cl::Kernel mConvKernel;
    cl::Kernel mHoughKernel;

    std::unique_ptr< Ocl::DataBuffer<Ocl::Pos> > mTempData;

    static const char sSource[];
};

};

