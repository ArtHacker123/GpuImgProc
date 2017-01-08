#pragma once

#include "OclDataBuffer.h"
#include <memory>

namespace Ogl
{

class HistogramPainterPrv;

class HistogramPainter
{
public:
    HistogramPainter(const cl::Context& ctxt);
    ~HistogramPainter();

    void setColor(float red, float green, float blue);
    void draw(const cl::CommandQueue& queue, const Ocl::DataBuffer<int>& histData, int maxValue);

private:
    std::unique_ptr<Ogl::HistogramPainterPrv> mPrv;
};

};
