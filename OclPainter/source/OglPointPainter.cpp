#include "OglPointPainter.h"
#include "OglPointPainterPrv.h"

using namespace Ogl;

PointPainter::PointPainter(cl::Context& ctxt, cl::CommandQueue& queue, size_t maxPoints)
    :mPrv(new Ogl::PointPainterPrv(ctxt, queue, maxPoints))
{
}

PointPainter::~PointPainter()
{
}

void PointPainter::draw(Ocl::DataBuffer<Ocl::Pos>& points, size_t count, size_t width, size_t height)
{
    mPrv->draw(points, count, width, height);
}
