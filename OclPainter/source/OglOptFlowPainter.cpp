#include "OglOptFlowPainter.h"
#include "OglOptFlowPainterPrv.h"

using namespace Ogl;

OptFlowPainter::OptFlowPainter(const cl::Context& ctxt, size_t maxSize)
    :mPrv(new Ogl::OptFlowPainterPrv(ctxt, maxSize))
{
}

OptFlowPainter::~OptFlowPainter()
{
}

void OptFlowPainter::draw(const cl::CommandQueue& queue, Ocl::DataBuffer<Ocl::OptFlowData>& fd, size_t count, size_t width, size_t height)
{
    mPrv->draw(queue, fd, count, width, height);
}
