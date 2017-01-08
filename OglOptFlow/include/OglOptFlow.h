#pragma once

#include "OglImage.h"
#include "OclDataTypes.h"
#include "OclDataBuffer.h"

#include <memory>
#include <vector>

namespace Ogl
{

class OptFlowPrv;

class OptFlow
{
public:
    OptFlow(const cl::Context& ctxt, GLsizei levels);
    ~OptFlow();

    bool process(const cl::CommandQueue& queue, Ocl::DataBuffer<Ocl::OptFlowData>& fd, size_t& outCount, const Ogl::Image<GL_RED>& currImg, const Ogl::Image<GL_RED>& prevImg, GLfloat rvalue, GLfloat minFlowDist);

private:
    std::unique_ptr<OptFlowPrv> mPrv;
};

};
