#include "OglOptFlowPrv.h"

using namespace Ogl;

OptFlow::OptFlow(cl::Context& ctxt, cl::CommandQueue& queue, GLsizei levels)
    :mPrv(new Ogl::OptFlowPrv(ctxt, queue, levels))
{
}

OptFlow::~OptFlow()
{
}

void OptFlow::process(Ocl::DataBuffer<Ocl::OptFlowData>& flowData, size_t& outCount, const Ogl::Image<GL_RED>& currImg, const Ogl::Image<GL_RED>& prevImg, GLfloat rvalue, GLfloat minFlowDist)
{
    mPrv->process(flowData, outCount, currImg, prevImg, rvalue, minFlowDist);
}

