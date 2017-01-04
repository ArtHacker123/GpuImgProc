#pragma once

#include "OclDataTypes.h"
#include "OclDataBuffer.h"
#include <memory>

namespace Ocl
{

class HoughLinesPrv;

class HoughLines
{
public:
    HoughLines(cl::Context& ctxt, cl::CommandQueue& queue);
    ~HoughLines();

    size_t process(const cl::Image2D& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, size_t& houghCount);
    size_t process(const cl::ImageGL& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, size_t& houghCount);

private:
    std::unique_ptr<Ocl::HoughLinesPrv> mPrv;
};

};
