#include "OclCompact.h"
#include "OclCompactPrv.h"

using namespace Ocl;

Compact::Compact(const cl::Context& ctxt)
    :mPrv(new Ocl::CompactPrv(ctxt))
{
}

Compact::~Compact()
{
}

size_t Compact::process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<cl_int2>& coords, float threshold, size_t& count)
{
    return mPrv->process(queue, inpImage, coords, threshold, count);
}

size_t Compact::process_cartesian(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<cl_int2>& coords, float threshold, size_t& count)
{
    return mPrv->process_cartesian(queue, inpImage, coords, threshold, count);
}

size_t Compact::process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::OptFlowData>& flowData, float threshold, size_t& count)
{
    return mPrv->process(queue, inpImage, flowData, threshold, count);
}

size_t Compact::process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::HoughData>& houghData, size_t threshold, size_t& count)
{
    return mPrv->process(queue, inpImage, houghData, threshold, count);
}
