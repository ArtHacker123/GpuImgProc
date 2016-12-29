#include "OglOptFlowPainter.h"
#include "OglOptFlowPainterPrv.h"

using namespace Ogl;

OptFlowPainter::OptFlowPainter(cl::Context& ctxt, cl::CommandQueue& queue, size_t maxSize)
    :mPrv(new Ogl::OptFlowPainterPrv(ctxt, queue, maxSize))
{
}

OptFlowPainter::~OptFlowPainter()
{
}

void OptFlowPainter::draw(Ocl::DataBuffer<Ocl::OptFlowData>& fd, size_t count, size_t width, size_t height)
{
    mPrv->draw(fd, count, width, height);
}
