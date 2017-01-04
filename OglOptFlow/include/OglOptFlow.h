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
    OptFlow(cl::Context& ctxt, cl::CommandQueue& queue, GLsizei levels);
    ~OptFlow();

    bool process(Ocl::DataBuffer<Ocl::OptFlowData>& flowData, size_t& outCount, const Ogl::Image<GL_RED>& currImg, const Ogl::Image<GL_RED>& prevImg, GLfloat rvalue, GLfloat minFlowDist);

private:
    std::unique_ptr<OptFlowPrv> mPrv;
};

};
