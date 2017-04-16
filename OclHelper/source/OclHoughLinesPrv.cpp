#include "OclHoughLinesPrv.h"
#include "OclUtils.h"
#include <sstream>

using namespace Ocl;

size_t comp_rho(size_t x, size_t y)
{
    size_t rho = (1 + (size_t)ceil(0.5*sqrt((x*x)+(y*y))));
    rho = (rho/32)+(((rho%32)==0)?0:1);
    return (32*rho);
}

HoughLinesPrv::HoughLinesPrv(const cl::Context& ctxt)
    :mRho(0),
     mContext(ctxt),
     mEdgeCount(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 1),
     mCompact(mContext)
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

    mNmsKernel = cl::Kernel(mPgm, "non_max_suppress");
    mHoughKernel = cl::Kernel(mPgm, "hough_line_transform");

    mWaitEvent[0].resize(1);
    mWaitEvent[1].resize(1);
    mWaitEvent[2].resize(1);
}

void HoughLinesPrv::createTempBuffers(const cl::Image& inpImage)
{
    size_t w = 0, h = 0;
    inpImage.getImageInfo<size_t>(CL_IMAGE_WIDTH, &w);
    inpImage.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &h);

    size_t maxEdges = (w*h) >> 2;
    size_t count = (mEdgeData.get() == 0)?0:mEdgeData->count();
    if (count < maxEdges)
    {
        mEdgeData.reset(new Ocl::DataBuffer<cl_int2>(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, maxEdges));
    }

    size_t imgRho = 0;
    mRho = comp_rho(w, h);
    if (mHoughImg.get() != 0)
    {
        mHoughImg->getImageInfo<size_t>(CL_IMAGE_WIDTH, &imgRho);
    }

    if (imgRho != mRho)
    {
        mHoughImg.reset(new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_UNSIGNED_INT32), mRho, 360));
        mHoughNmsImg.reset(new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_UNSIGNED_INT32), mRho, 360));
    }
}

void HoughLinesPrv::computeHLT(const cl::CommandQueue& queue, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& events)
{
    size_t shMemSize = mRho*sizeof(int);
    mHoughKernel.setArg(0, mEdgeData->buffer());
    mHoughKernel.setArg(1, count.buffer());
    mHoughKernel.setArg(2, (int)mRho);
    mHoughKernel.setArg(3, *mHoughImg);
    mHoughKernel.setArg(4, shMemSize, 0);
    mWaitEvent[0][0] = events.back();
    events.resize(events.size()+1);
    queue.enqueueNDRangeKernel(mHoughKernel, cl::NullRange, cl::NDRange(360*256), cl::NDRange(256), &mWaitEvent[0], &events.back());
}

void HoughLinesPrv::nonMaxSuppress(const cl::CommandQueue& queue, size_t threshold, std::vector<cl::Event>& events)
{
    mNmsKernel.setArg(0, *mHoughImg);
    mNmsKernel.setArg(1, *mHoughNmsImg);
    mNmsKernel.setArg(2, (int)threshold);
    size_t width = (mRho/8)+(((mRho%8)==0)?0:1);
    mWaitEvent[1][0] = events.back();
    events.resize(events.size()+1);
    queue.enqueueNDRangeKernel(mNmsKernel, cl::NullRange, cl::NDRange(8*width, 360), cl::NDRange(8, 8), &mWaitEvent[1], &events.back());
}

void HoughLinesPrv::process(const cl::CommandQueue& queue, const cl::Image2D& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, Ocl::DataBuffer<cl_int>& houghCount,
    std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent)
{
    createTempBuffers(inpImage);
    mCompact.process_cartesian(queue, inpImage, *mEdgeData, 1.0, mEdgeCount, events, pWaitEvent);
    computeHLT(queue, mEdgeCount, events);
    nonMaxSuppress(queue, minSize, events);
    mWaitEvent[2][0] = events.back();
    mCompact.process(queue, *mHoughNmsImg, hData, (int)minSize, houghCount, events, &mWaitEvent[2]);
}

void HoughLinesPrv::process(const cl::CommandQueue& queue, const cl::ImageGL& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, Ocl::DataBuffer<cl_int>& houghCount,
    std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent)
{
    createTempBuffers(inpImage);
    std::vector<cl::Memory> gl_objs = { inpImage };
    queue.enqueueAcquireGLObjects(&gl_objs);
    mCompact.process_cartesian(queue, inpImage, *mEdgeData, 1.0, mEdgeCount, events, pWaitEvent);
    queue.enqueueReleaseGLObjects(&gl_objs);
    computeHLT(queue, mEdgeCount, events);
    nonMaxSuppress(queue, minSize, events);
    mWaitEvent[2][0] = events.back();
    mCompact.process(queue, *mHoughNmsImg, hData, (int)minSize, houghCount, events, &mWaitEvent[2]);
}
