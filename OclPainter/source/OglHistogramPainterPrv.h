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
    HistogramPainterPrv(const cl::Context& ctxt);
    ~HistogramPainterPrv();

    void setColor(GLfloat r, GLfloat g, GLfloat b);
    void draw(const cl::CommandQueue& queue, const Ocl::DataBuffer<int>& hData, int maxValue);

private:
    void compute(const cl::CommandQueue& queue, const Ocl::DataBuffer<int>& rgbBins, int maxValue);

private:
    const cl::Context& mContext;

    Ogl::Buffer mBuffer;
    cl::BufferGL mBufferGL;
    Ogl::Painter<Ogl::ColorShader> mPainter;

    cl::Kernel mKernel;
    cl::Program mProgram;

    static const char source[];
};

};
