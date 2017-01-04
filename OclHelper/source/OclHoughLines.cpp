#include "OclHoughLines.h"
#include "OclHoughLinesPrv.h"

using namespace Ocl;

#define COMPUTE_RHO(x, y) (1+(size_t)ceil(0.5*sqrt((x*x)+(y*y))))

HoughData::HoughData(cl::Context& ctxt, cl::CommandQueue& queue, size_t width, size_t height)
    :mCtxt(ctxt),
     mQueue(queue),
     mImage(mCtxt, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_UNSIGNED_INT32), COMPUTE_RHO(width, height), 360)
{
}

HoughData::~HoughData()
{
}

HoughLines::HoughLines(cl::Context& ctxt, cl::CommandQueue& queue)
    :mPrv(new Ocl::HoughLinesPrv(ctxt, queue))
{
}

HoughLines::~HoughLines()
{
}

size_t HoughLines::process(const Ocl::DataBuffer<Ocl::Pos>& edgeData, size_t edgeCount, size_t width, size_t height, Ocl::HoughData& hData)
{
    return mPrv->process(edgeData, edgeCount, width, height, hData);
}
