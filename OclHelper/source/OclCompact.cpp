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

void Compact::process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<cl_int2>& coords,
                    float threshold, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& event, std::vector<cl::Event>* pWaitEvent)
{
    mPrv->process(queue, inpImage, coords, threshold, count, event, pWaitEvent);
}

void Compact::process_cartesian(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<cl_int2>& coords,
                     float threshold, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& event, std::vector<cl::Event>* pWaitEvent)
{
    mPrv->process_cartesian(queue, inpImage, coords, threshold, count, event, pWaitEvent);
}

void Compact::process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::OptFlowData>& flowData, 
                        float threshold, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& event, std::vector<cl::Event>* pWaitEvent)
{
    mPrv->process(queue, inpImage, flowData, threshold, count, event, pWaitEvent);
}

void Compact::process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::HoughData>& houghData,
                        size_t threshold, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& event, std::vector<cl::Event>* pWaitEvent)
{
    mPrv->process(queue, inpImage, houghData, threshold, count, event, pWaitEvent);
}
