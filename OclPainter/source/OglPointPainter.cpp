#include "OglPointPainter.h"
#include "OglPointPainterPrv.h"

using namespace Ogl;

PointPainter::PointPainter(const cl::Context& ctxt, size_t maxPoints)
    :mPrv(new Ogl::PointPainterPrv(ctxt, maxPoints))
{
}

PointPainter::~PointPainter()
{
}

void PointPainter::draw(const cl::CommandQueue& queue, Ocl::DataBuffer<cl_int2>& points, size_t count, size_t width, size_t height)
{
    mPrv->draw(queue, points, count, width, height);
}
