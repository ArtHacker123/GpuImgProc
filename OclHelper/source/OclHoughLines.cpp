#include "OclHoughLines.h"
#include "OclHoughLinesPrv.h"

using namespace Ocl;

HoughLines::HoughLines(const cl::Context& ctxt)
    :mPrv(new Ocl::HoughLinesPrv(ctxt))
{
}

HoughLines::~HoughLines()
{
}

void HoughLines::process(const cl::CommandQueue& queue, const cl::Image2D& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, Ocl::DataBuffer<cl_int>& houghCount,
                    std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent)
{
    mPrv->process(queue, inpImage, minSize, hData, houghCount, events, pWaitEvent);
}

void HoughLines::process(const cl::CommandQueue& queue, const cl::ImageGL& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, Ocl::DataBuffer<cl_int>& houghCount,
                    std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent)
{
    mPrv->process(queue, inpImage, minSize, hData, houghCount, events, pWaitEvent);
}
