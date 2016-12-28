#pragma once

#include "OglBuffer.h"
#include "OglPainter.h"
#include "OglColorShader.h"

#include "DataTypes.h"
#include "DataBuffer.h"

#include <CL/cl.hpp>

class OptFlowPainter
{
public:
    OptFlowPainter(cl::Context& ctxt, cl::CommandQueue& queue, size_t maxSize);
    ~OptFlowPainter();

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
