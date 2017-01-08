#pragma once

#include "OglBuffer.h"
#include "OglPainter.h"
#include "OglColorShader.h"

#include "OclDataTypes.h"
#include "OclDataBuffer.h"

namespace Ogl
{

class OptFlowPainterPrv
{
public:
    OptFlowPainterPrv(const cl::Context& ctxt, size_t maxSize);
    ~OptFlowPainterPrv();

public:
    void draw(const cl::CommandQueue& queue, Ocl::DataBuffer<Ocl::OptFlowData>& fd, size_t count, size_t width, size_t height);

private:
    size_t mMaxSize;
    const cl::Context& mContext;

    Ogl::Buffer mOptFlowBuff;
    Ogl::Painter<Ogl::ColorShader> mPainter;

    cl::Program mPgm;
    cl::Kernel mKernel;
    static const char sSource[];
};

};
