#pragma once

#include "OclDataBuffer.h"
#include <memory>

namespace Ogl
{

class HistogramPainterPrv;

class HistogramPainter
{
public:
    HistogramPainter(cl::Context& ctxt, cl::CommandQueue& queue);
    ~HistogramPainter();

    void setColor(float red, float green, float blue);
    void draw(const Ocl::DataBuffer<int>& histData, int maxValue);

private:
    std::unique_ptr<Ogl::HistogramPainterPrv> mPrv;
};

};
