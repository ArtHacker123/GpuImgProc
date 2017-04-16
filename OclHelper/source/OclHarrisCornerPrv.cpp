#include "OclHarrisCornerPrv.h"
#include "OclUtils.h"
#include <sstream>

using namespace Ocl;

HarrisCornerPrv::HarrisCornerPrv(const cl::Context& ctxt)
    :mWidth(0),
     mHeight(0),
     mLocSizeX(16),
     mLocSizeY(8),
     mContext(ctxt),
     mCompact(ctxt),
     mWlistCount(0)
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

HarrisCornerPrv::~HarrisCornerPrv()
{
}


void HarrisCornerPrv::init()
{
    cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
    mPgm = cl::Program(mContext, source);
    
    std::vector<cl::Device> devices = mContext.getInfo<CL_CONTEXT_DEVICES>();
    std::string version;
    devices[0].getInfo<std::string>(CL_DEVICE_OPENCL_C_VERSION, &version);

    std::ostringstream options;
    options << " -DBLK_SIZE_X=" << mLocSizeX << " -DBLK_SIZE_Y=" << mLocSizeY;
    mPgm.build(options.str().c_str());

    mGradKernel = cl::Kernel(mPgm, "gradient");
    mEigenKernel = cl::Kernel(mPgm, "eigen");
    mCornerKernel = cl::Kernel(mPgm, "nms");
    
    mWaitList.resize(5);
    for (size_t i = 0; i < 5; i++)
    {
        mWaitList[i].resize(1);
    }
}

void HarrisCornerPrv::createIntImages(const cl::Image& inpImg)
{
    size_t w = 0, h = 0;
    inpImg.getImageInfo<size_t>(CL_IMAGE_WIDTH, &mWidth);
    inpImg.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &mHeight);

    if (mIxIyImg.get() != 0)
    {
        mIxIyImg->getImageInfo<size_t>(CL_IMAGE_WIDTH, &w);
        mIxIyImg->getImageInfo<size_t>(CL_IMAGE_HEIGHT, &h);
    }

    if (w != mWidth || h != mHeight)
    {
        mIxIyImg.reset(new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), mWidth, mHeight));
        mEigenImg.reset(new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), mWidth, mHeight));
        mCornerImg.reset(new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), mWidth, mHeight));
    }
}

void HarrisCornerPrv::gradient(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent)
{
    mGradKernel.setArg(0, inpImg);
    mGradKernel.setArg(1, outImg);
    events.resize(events.size()+1);
    queue.enqueueNDRangeKernel(mGradKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), pWaitEvent, &events.back());
}

void HarrisCornerPrv::eigen(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, std::vector<cl::Event>& events)
{
    mEigenKernel.setArg(0, inpImg);
    mEigenKernel.setArg(1, outImg);
    mWaitList[mWlistCount][0] = events.back();
    events.resize(events.size()+1);
    queue.enqueueNDRangeKernel(mEigenKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), &mWaitList[mWlistCount], &events.back());
    ++mWlistCount;
}

void HarrisCornerPrv::suppress(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, float value, std::vector<cl::Event>& events)
{
    mCornerKernel.setArg(0, inpImg);
    mCornerKernel.setArg(1, outImg);
    mCornerKernel.setArg(2, value);
    mWaitList[mWlistCount][0] = events.back();
    events.resize(events.size()+1);
    queue.enqueueNDRangeKernel(mCornerKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), &mWaitList[mWlistCount], &events.back());
    ++mWlistCount;
}

void HarrisCornerPrv::process(const cl::CommandQueue& queue, const cl::Image2D& inpImg, Ocl::DataBuffer<cl_int2>& corners, float value, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent)
{
    mWlistCount = 0;
    createIntImages(inpImg);
    gradient(queue, inpImg, *mIxIyImg, events, pWaitEvent);
    eigen(queue, *mIxIyImg, *mEigenImg, events);
    suppress(queue, *mEigenImg, *mCornerImg, value, events);
    mWaitList[mWlistCount][0] = events.back();
    mCompact.process(queue, *mCornerImg, corners, 1.0f, count, events, &mWaitList[mWlistCount]);
}

void HarrisCornerPrv::process(const cl::CommandQueue& queue, const cl::ImageGL& inpImg, Ocl::DataBuffer<cl_int2>& corners, float value, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent)
{
    mWlistCount = 0;
    createIntImages(inpImg);
    std::vector<cl::Memory> gl_objs = { inpImg };
    queue.enqueueAcquireGLObjects(&gl_objs);
    gradient(queue, inpImg, *mIxIyImg, events, pWaitEvent);
    queue.enqueueReleaseGLObjects(&gl_objs);
    eigen(queue, *mIxIyImg, *mEigenImg, events);
    suppress(queue, *mEigenImg, *mCornerImg, value, events);
    mWaitList[mWlistCount][0] = events.back();
    mCompact.process(queue, *mCornerImg, corners, 1.0f, count, events, &mWaitList[mWlistCount]);
}
