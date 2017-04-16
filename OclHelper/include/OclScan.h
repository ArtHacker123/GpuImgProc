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

    void process(const cl::CommandQueue& queue, Ocl::DataBuffer<int>& buffer,
                   std::vector<cl::Event>& event, std::vector<cl::Event>* pWaitEvent = 0);

private:
    std::unique_ptr<Ocl::ScanPrv> mPrv;
};

};
