#pragma once

#include "OclDataBuffer.h"
#include <memory>

namespace Ocl
{

class ScanPrv;

class Scan
{
public:
    Scan(cl::Context& ctxt, cl::CommandQueue& queue);
    ~Scan();

    size_t process(Ocl::DataBuffer<int>& buffer);

private:
    std::unique_ptr<Ocl::ScanPrv> mPrv;
};

};
