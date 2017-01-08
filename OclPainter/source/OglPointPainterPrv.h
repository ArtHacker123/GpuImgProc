#pragma once

#include "OglBuffer.h"
#include "OglPainter.h"
#include "OglColorShader.h"

#include "OclDataTypes.h"
#include "OclDataBuffer.h"

namespace Ogl
{

class PointPainterPrv
{
public:
    PointPainterPrv(const cl::Context& ctxt, size_t maxPoints);
    ~PointPainterPrv();

public:
    void draw(const cl::CommandQueue& queue, Ocl::DataBuffer<Ocl::Pos>& points, size_t count, size_t width, size_t height);

private:
	size_t mMaxPoints;
    const cl::Context& mContext;

	Ogl::Buffer mPointBuff;
    Ogl::Painter<Ogl::ColorShader> mPainter;

    cl::Program mPgm;
    cl::Kernel mKernel;
    static const char sSource[];
};

}
