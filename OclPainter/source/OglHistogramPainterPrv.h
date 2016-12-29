#pragma once

#include "glew.h"
#include <GL/GL.h>
#include "OglBuffer.h"
#include "OglPainter.h"
#include "OglColorShader.h"

#include <CL/cl.hpp>
#include "OclDataBuffer.h"

namespace Ogl
{

class HistogramPainterPrv
{
public:
    HistogramPainterPrv(cl::Context& ctxt, cl::CommandQueue& queue);
    ~HistogramPainterPrv();

    void setColor(GLfloat r, GLfloat g, GLfloat b);
    void draw(const Ocl::DataBuffer<int>& hData, int maxValue);

private:
    void compute(const Ocl::DataBuffer<int>& rgbBins, int maxValue);

private:
    cl::Context& mContext;
    cl::CommandQueue& mQueue;

    Ogl::Buffer mBuffer;
    cl::BufferGL mBufferGL;
    Ogl::Painter<Ogl::ColorShader> mPainter;

    cl::Kernel mKernel;
    cl::Program mProgram;

    static const char source[];
};

};
