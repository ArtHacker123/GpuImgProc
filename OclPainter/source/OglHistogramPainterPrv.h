#pragma once

#include "glew.h"
#include <GL/GL.h>
#include "OglBuffer.h"
#include "OglPainter.h"
#include "OglColorShader.h"
#include "IOglGeometry.h"

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
    void draw(const cl::CommandQueue& queue, const Ocl::DataBuffer<cl_int>& hData, int maxValue);
    void draw(const Ogl::IGeometry::Rect& vp, const cl::CommandQueue& queue, const Ocl::DataBuffer<cl_int>& histData, int maxValue);

private:
    void compute(const cl::CommandQueue& queue, const Ocl::DataBuffer<cl_int>& rgbBins, int maxValue);

private:
    const cl::Context& mContext;

    Ogl::Buffer mBuffer;
    cl::BufferGL mBufferGL;
    Ogl::Painter<Ogl::ColorShader> mPainter;

    cl::Kernel mKernel;
    cl::Program mProgram;

    static const char sSource[];
};

};
