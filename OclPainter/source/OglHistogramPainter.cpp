#include "OglHistogramPainter.h"
#include "OglHistogramPainterPrv.h"

namespace Ogl
{

HistogramPainter::HistogramPainter(cl::Context& ctxt, cl::CommandQueue& queue)
    :mPrv(new Ogl::HistogramPainterPrv(ctxt, queue))
{
}

HistogramPainter::~HistogramPainter()
{
}

void HistogramPainter::setColor(float red, float green, float blue)
{
    mPrv->setColor(red, green, blue);
}

void HistogramPainter::draw(const Ocl::DataBuffer<int>& histData, int maxValue)
{
    mPrv->draw(histData, maxValue);
}

}
