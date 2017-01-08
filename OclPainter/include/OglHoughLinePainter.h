#pragma once

#include "OclDataTypes.h"
#include "OclDataBuffer.h"
#include <memory>

namespace Ogl
{

class HoughLinePainterPrv;

class HoughLinePainter
{
public:
    HoughLinePainter(const cl::Context& ctxt, size_t maxSize);
    ~HoughLinePainter();

    void draw(const cl::CommandQueue& queue, Ocl::DataBuffer<Ocl::HoughData>& hData, size_t count, size_t width, size_t height);

private:
    std::unique_ptr<Ogl::HoughLinePainterPrv> mPrv;
};

};
