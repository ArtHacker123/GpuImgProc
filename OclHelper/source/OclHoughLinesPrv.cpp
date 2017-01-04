#include "OclHoughLinesPrv.h"
#include "OclUtils.h"
#include <sstream>

using namespace Ocl;

HoughLinesPrv::HoughLinesPrv(cl::Context& ctxt, cl::CommandQueue& queue)
    :mContext(ctxt),
     mQueue(queue)
{
    try
    {
        init();
    }

    catch (cl::Error error)
    {
        fprintf(stderr, "%s", error.what());
        exit(0);
    }
}

HoughLinesPrv::~HoughLinesPrv()
{
}

void HoughLinesPrv::init()
{
    cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
    mPgm = cl::Program(mContext, source);
    mPgm.build();

    mConvKernel = cl::Kernel(mPgm, "convert_coords");
    mHoughKernel = cl::Kernel(mPgm, "hough_line_transform");
}

void HoughLinesPrv::createTempData(const Ocl::DataBuffer<Ocl::Pos>& edgeData)
{
    size_t count = (mTempData.get() == 0)?0:mTempData->count();
    if (count < edgeData.count())
    {
        mTempData.reset(new Ocl::DataBuffer<Ocl::Pos>(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, edgeData.count()));
    }
}

size_t HoughLinesPrv::process(const Ocl::DataBuffer<Ocl::Pos>& edgeData, size_t edgeCount, size_t width, size_t height, Ocl::HoughData& hData)
{
    cl::Event event;
    createTempData(edgeData);

    mConvKernel.setArg(0, edgeData.buffer());
    mConvKernel.setArg(1, mTempData->buffer());
    mConvKernel.setArg(2, (int)edgeCount);
    mConvKernel.setArg(3, (int)(width/2));
    mConvKernel.setArg(4, (int)(height/2));
    size_t gSize = (edgeCount/16)+((edgeCount%16)?1:0);
    gSize *= 16;
    mQueue.enqueueNDRangeKernel(mConvKernel, cl::NullRange, cl::NDRange(gSize), cl::NDRange(16), NULL, &event);
    event.wait();
    size_t time = kernelExecTime(mQueue, event);

    size_t maxRho = 1+ceil(0.5*sqrt((double)((width*width)+(height*height))));
    size_t shMemSize = maxRho*sizeof(int);
    mHoughKernel.setArg(0, mTempData->buffer());
    mHoughKernel.setArg(1, (int)edgeCount);
    mHoughKernel.setArg(2, (int)maxRho);
    mHoughKernel.setArg(3, hData.getImage());
    mHoughKernel.setArg(4, shMemSize, 0);
    mQueue.enqueueNDRangeKernel(mHoughKernel, cl::NullRange, cl::NDRange(360*256), cl::NDRange(256), NULL, &event);
    event.wait();
    time += kernelExecTime(mQueue, event);

    return time;
}

