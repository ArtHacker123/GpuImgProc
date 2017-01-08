#pragma once

#include "OclDataBuffer.h"
#include <memory>

namespace Ocl
{

class ScanPrv;

class Scan
{
public:
    Scan(const cl::Context& ctxt);
    ~Scan();

    size_t process(const cl::CommandQueue& queue, Ocl::DataBuffer<int>& buffer);

private:
    std::unique_ptr<Ocl::ScanPrv> mPrv;
};

};
