#pragma once

#include "OclDataTypes.h"
#include "OclDataBuffer.h"
#include <memory>

namespace Ogl
{

class OptFlowPainterPrv;

class OptFlowPainter
{
public:
    OptFlowPainter(const cl::Context& ctxt, size_t maxSize);
    ~OptFlowPainter();

    void draw(const cl::CommandQueue& queue, Ocl::DataBuffer<Ocl::OptFlowData>& fd, size_t count, size_t width, size_t height);

private:
    std::unique_ptr<Ogl::OptFlowPainterPrv> mPrv;
};

};
