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
    PointPainterPrv(cl::Context& ctxt, cl::CommandQueue& queue, size_t maxPoints);
    ~PointPainterPrv();

public:
    void draw(Ocl::DataBuffer<Ocl::Pos>& points, size_t count, size_t width, size_t height);

private:
	size_t mMaxPoints;
    cl::Context& mContext;
    cl::CommandQueue mQueue;

	Ogl::Buffer mPointBuff;
    Ogl::Painter<Ogl::ColorShader> mPainter;

    cl::Program mPgm;
    cl::Kernel mKernel;
    static const char sSource[];
};

}
