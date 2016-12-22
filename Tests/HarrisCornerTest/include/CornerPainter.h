#pragma once

#include "OglBuffer.h"
#include "OglPainter.h"
#include "OglColorShader.h"

#include "DataTypes.h"
#include "DataBuffer.h"

#include <CL/cl.hpp>

class CornerPainter
{
public:
    CornerPainter(cl::Context& ctxt, cl::CommandQueue& queue, size_t maxCorners);
    ~CornerPainter();

public:
    void draw(Ocl::DataBuffer<Ocl::Pos>& corners, size_t width, size_t height, size_t count);

private:
	size_t mMaxCorners;
    cl::Context& mContext;
    cl::CommandQueue mQueue;

	Ogl::Buffer mCornerBuff;
    Ogl::Painter<Ogl::ColorShader> mPainter;

    cl::Program mPgm;
    cl::Kernel mKernel;
    static const char sSource[];
};
