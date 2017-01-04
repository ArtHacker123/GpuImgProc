#pragma once

#include "OclDataTypes.h"
#include "OclDataBuffer.h"
#include <memory>

namespace Ocl
{

class HoughLinesPrv;

class HoughData
{
public:
    HoughData(cl::Context& ctxt, cl::CommandQueue& queue, size_t width, size_t height);
    ~HoughData();

    cl::Image2D& getImage() { return mImage; };

private:
    cl::Context mCtxt;
    cl::CommandQueue mQueue;

    cl::Image2D mImage;
};

class HoughLines
{
public:
    HoughLines(cl::Context& ctxt, cl::CommandQueue& queue);
    ~HoughLines();

    size_t process(const Ocl::DataBuffer<Ocl::Pos>& edgeData, size_t edgeCount, size_t width, size_t height, Ocl::HoughData& hData);

private:
    std::unique_ptr<Ocl::HoughLinesPrv> mPrv;
};

};

