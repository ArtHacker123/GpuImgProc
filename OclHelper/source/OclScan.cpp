#include "OclScan.h"
#include "OclScanPrv.h"

using namespace Ocl;

Scan::Scan(const cl::Context& ctxt)
    :mPrv(new Ocl::ScanPrv(ctxt))
{
}

Scan::~Scan()
{
}

size_t Scan::process(const cl::CommandQueue& queue, Ocl::DataBuffer<int>& buffer)
{
    return mPrv->process(queue, buffer);
}
