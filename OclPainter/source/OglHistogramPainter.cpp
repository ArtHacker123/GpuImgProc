#include "OglHistogramPainter.h"
#include "OglHistogramPainterPrv.h"

namespace Ogl
{

HistogramPainter::HistogramPainter(const cl::Context& ctxt)
    :mPrv(new Ogl::HistogramPainterPrv(ctxt))
{
}

HistogramPainter::~HistogramPainter()
{
}

void HistogramPainter::setColor(float red, float green, float blue)
{
    mPrv->setColor(red, green, blue);
}

void HistogramPainter::draw(const cl::CommandQueue& queue, const Ocl::DataBuffer<cl_int>& histData, int maxValue)
{
    mPrv->draw(queue, histData, maxValue);
}

void HistogramPainter::draw(const Ogl::IGeometry::Rect& vp, const cl::CommandQueue& queue, const Ocl::DataBuffer<cl_int>& histData, int maxValue)
{
    mPrv->draw(vp, queue, histData, maxValue);
}

}
