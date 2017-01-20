#pragma once

#include "OclDataTypes.h"
#include "OclDataBuffer.h"
#include <memory>

namespace Ogl
{

class PointPainterPrv;

class PointPainter
{
public:
    PointPainter(const cl::Context& ctxt, size_t maxPoints);
    ~PointPainter();

    void draw(const cl::CommandQueue& queue, Ocl::DataBuffer<cl_int2>& points, size_t count, size_t width, size_t height);

private:
    std::unique_ptr<Ogl::PointPainterPrv> mPrv;
};

}
