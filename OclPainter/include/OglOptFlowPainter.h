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
    OptFlowPainter(cl::Context& ctxt, cl::CommandQueue& queue, size_t maxSize);
    ~OptFlowPainter();

    void draw(Ocl::DataBuffer<Ocl::OptFlowData>& flowData, size_t count, size_t width, size_t height);

private:
    std::unique_ptr<Ogl::OptFlowPainterPrv> mPrv;
};

};
