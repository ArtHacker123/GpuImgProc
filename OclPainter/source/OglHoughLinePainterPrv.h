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
    HoughLinePainterPrv(const cl::Context& ctxt, size_t maxSize);
    ~HoughLinePainterPrv();

public:
    void draw(const cl::CommandQueue& queue, Ocl::DataBuffer<Ocl::HoughData>& flowData, size_t count, size_t width, size_t height);

private:
    size_t mMaxSize;
    const cl::Context& mContext;

    Ogl::Buffer mHoughLineBuff;
    Ogl::Painter<Ogl::ColorShader> mPainter;

    cl::Program mPgm;
    cl::Kernel mKernel;
    static const char sSource[];
};

};
