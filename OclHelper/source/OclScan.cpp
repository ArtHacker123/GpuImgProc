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

void Scan::process(const cl::CommandQueue& queue, Ocl::DataBuffer<int>& buffer,
                    std::vector<cl::Event>& event, std::vector<cl::Event>* pWaitEvent)
{
    mPrv->process(queue, buffer, event, pWaitEvent);
}
