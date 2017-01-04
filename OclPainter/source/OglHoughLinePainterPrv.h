#pragma once

#include "OglBuffer.h"
#include "OglPainter.h"
#include "OglColorShader.h"

#include "OclDataTypes.h"
#include "OclDataBuffer.h"

namespace Ogl
{

class HoughLinePainterPrv
{
public:
    HoughLinePainterPrv(cl::Context& ctxt, cl::CommandQueue& queue, size_t maxSize);
    ~HoughLinePainterPrv();

public:
    void draw(Ocl::DataBuffer<Ocl::HoughData>& flowData, size_t count, size_t width, size_t height);

private:
    size_t mMaxSize;
    cl::Context& mContext;
    cl::CommandQueue mQueue;

    Ogl::Buffer mHoughLineBuff;
    Ogl::Painter<Ogl::ColorShader> mPainter;

    cl::Program mPgm;
    cl::Kernel mKernel;
    static const char sSource[];
};

};
