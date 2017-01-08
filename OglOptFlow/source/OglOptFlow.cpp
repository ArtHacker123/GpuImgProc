#include "OglOptFlowPrv.h"

using namespace Ogl;

OptFlow::OptFlow(const cl::Context& ctxt, GLsizei levels)
    :mPrv(new Ogl::OptFlowPrv(ctxt, levels))
{
}

OptFlow::~OptFlow()
{
}

bool OptFlow::process(const cl::CommandQueue& queue, Ocl::DataBuffer<Ocl::OptFlowData>& fd, size_t& outCount, const Ogl::Image<GL_RED>& currImg, const Ogl::Image<GL_RED>& prevImg, GLfloat rvalue, GLfloat minFlowDist)
{
    return mPrv->process(queue, fd, outCount, currImg, prevImg, rvalue, minFlowDist);
}

