#pragma once

#include "DataTypes.h"
#include "DataBuffer.h"
#include <memory>

namespace Ogl
{

class PointPainterPrv;

class PointPainter
{
public:
    PointPainter(cl::Context& ctxt, cl::CommandQueue& queue, size_t maxPoints);
    ~PointPainter();

    void draw(Ocl::DataBuffer<Ocl::Pos>& points, size_t count, size_t width, size_t height);

private:
    std::unique_ptr<Ogl::PointPainterPrv> mPrv;
};

}
