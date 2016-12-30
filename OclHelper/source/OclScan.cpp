#include "OclScan.h"
#include "OclScanPrv.h"

using namespace Ocl;

Scan::Scan(cl::Context& ctxt, cl::CommandQueue& queue)
	:mPrv(new Ocl::ScanPrv(ctxt, queue))
{
}

Scan::~Scan()
{
}

size_t Scan::process(Ocl::DataBuffer<int>& buffer)
{
	return mPrv->process(buffer);
}
