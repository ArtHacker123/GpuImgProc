#include "OclCompact.h"
#include "OclCompactPrv.h"

using namespace Ocl;

Compact::Compact(cl::Context& ctxt, cl::CommandQueue& queue)
    :mPrv(new Ocl::CompactPrv(ctxt, queue))
{
}

Compact::~Compact()
{
}

size_t Compact::process(const cl::Image& inpImage, Ocl::DataBuffer<Ocl::Pos>& coords, float threshold, size_t& count)
{
    return mPrv->process(inpImage, coords, threshold, count);
}

size_t Compact::process_cartesian(const cl::Image& inpImage, Ocl::DataBuffer<Ocl::Pos>& coords, float threshold, size_t& count)
{
    return mPrv->process_cartesian(inpImage, coords, threshold, count);
}

size_t Compact::process(const cl::Image& inpImage, Ocl::DataBuffer<Ocl::OptFlowData>& flowData, float threshold, size_t& count)
{
    return mPrv->process(inpImage, flowData, threshold, count);
}

size_t Compact::process(const cl::Image& inpImage, Ocl::DataBuffer<Ocl::HoughData>& houghData, size_t threshold, size_t& count)
{
    return mPrv->process(inpImage, houghData, threshold, count);
}
