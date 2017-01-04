#include "OglHoughLinePainter.h"
#include "OglHoughLinePainterPrv.h"

using namespace Ogl;

HoughLinePainter::HoughLinePainter(cl::Context& ctxt, cl::CommandQueue& queue, size_t maxSize)
    :mPrv(new Ogl::HoughLinePainterPrv(ctxt, queue, maxSize))
{
}

HoughLinePainter::~HoughLinePainter()
{
}

void HoughLinePainter::draw(Ocl::DataBuffer<Ocl::HoughData>& fd, size_t count, size_t width, size_t height)
{
    mPrv->draw(fd, count, width, height);
}
