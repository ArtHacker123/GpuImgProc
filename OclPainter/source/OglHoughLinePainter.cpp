#include "OglHoughLinePainter.h"
#include "OglHoughLinePainterPrv.h"

using namespace Ogl;

HoughLinePainter::HoughLinePainter(const cl::Context& ctxt, size_t maxSize)
    :mPrv(new Ogl::HoughLinePainterPrv(ctxt, maxSize))
{
}

HoughLinePainter::~HoughLinePainter()
{
}

void HoughLinePainter::draw(const cl::CommandQueue& queue, Ocl::DataBuffer<Ocl::HoughData>& hData, size_t count, size_t width, size_t height)
{
    mPrv->draw(queue, hData, count, width, height);
}
