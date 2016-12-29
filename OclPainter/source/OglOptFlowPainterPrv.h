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
    OptFlowPainterPrv(cl::Context& ctxt, cl::CommandQueue& queue, size_t maxSize);
    ~OptFlowPainterPrv();

public:
    void draw(Ocl::DataBuffer<Ocl::OptFlowData>& flowData, size_t count, size_t width, size_t height);

private:
    size_t mMaxSize;
    cl::Context& mContext;
    cl::CommandQueue mQueue;

    Ogl::Buffer mOptFlowBuff;
    Ogl::Painter<Ogl::ColorShader> mPainter;

    cl::Program mPgm;
    cl::Kernel mKernel;
    static const char sSource[];
};

};
