#include "OclCannyEdgePrv.h"
#include "OclUtils.h"
#include <sstream>

using namespace Ocl;

CannyEdgePrv::CannyEdgePrv(const cl::Context& ctxt)
    :mWidth(0),
     mHeight(0),
     mLocSizeX(16),
     mLocSizeY(8),
     mContext(ctxt)
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

CannyEdgePrv::~CannyEdgePrv()
{
}

void CannyEdgePrv::init()
{
    cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
    mPgm = cl::Program(mContext, source);
    mPgm.build();

    mGradKernel = cl::Kernel(mPgm, "gradient");
    mGaussKernel = cl::Kernel(mPgm, "gauss");
    mNmesKernel = cl::Kernel(mPgm, "nmes");
    mBinThreshKernel = cl::Kernel(mPgm, "binary_threshold");

    for (size_t i = 0; i < 3; i++)
    {
        mWaitEvent[i].resize(1);
    }
}

void CannyEdgePrv::createIntImages(const cl::Image& inpImg)
{
    size_t w = 0, h = 0;
    inpImg.getImageInfo<size_t>(CL_IMAGE_WIDTH, &mWidth);
    inpImg.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &mHeight);

    if (mGradImg.get() != 0)
    {
        mGradImg->getImageInfo<size_t>(CL_IMAGE_WIDTH, &w);
        mGradImg->getImageInfo<size_t>(CL_IMAGE_HEIGHT, &h);
    }

    if (w != mWidth || h != mHeight)
    {
        mGradImg.reset(new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), mWidth, mHeight));
        mGaussImg.reset(new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), mWidth, mHeight));
        mNmesImg.reset(new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), mWidth, mHeight));
    }

    //checkLocalGroupSizes();
}

void CannyEdgePrv::gradient(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, std::vector<cl::Event>& events)
{
    mGradKernel.setArg(0, inpImg);
    mGradKernel.setArg(1, outImg);
    mWaitEvent[0][0] = events.back();
    events.resize(events.size()+1);
    queue.enqueueNDRangeKernel(mGradKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), &mWaitEvent[0], &events.back());
}

void CannyEdgePrv::gauss(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent)
{
    mGaussKernel.setArg(0, inpImg);
    mGaussKernel.setArg(1, outImg);
    events.resize(events.size()+1);
    queue.enqueueNDRangeKernel(mGaussKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), pWaitEvent, &events.back());
}

void CannyEdgePrv::suppress(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, std::vector<cl::Event>& events)
{
    mNmesKernel.setArg(0, inpImg);
    mNmesKernel.setArg(1, outImg);
    mWaitEvent[1][0] = events.back();
    events.resize(events.size()+1);
    queue.enqueueNDRangeKernel(mNmesKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), &mWaitEvent[1], &events.back());
}

void CannyEdgePrv::checkLocalGroupSizes()
{
    mLocSizeX = Ocl::localGroupSize(mWidth);
    mLocSizeY = Ocl::localGroupSize(mHeight);
}

void CannyEdgePrv::binaryThreshold(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, float minThresh, float maxThresh, std::vector<cl::Event>& events)
{
    mBinThreshKernel.setArg(0, inpImg);
    mBinThreshKernel.setArg(1, outImg);
    mBinThreshKernel.setArg(2, minThresh);
    mBinThreshKernel.setArg(3, maxThresh);
    mWaitEvent[2][0] = events.back();
    events.resize(events.size()+1);
    queue.enqueueNDRangeKernel(mBinThreshKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), &mWaitEvent[2], &events.back());
}

void CannyEdgePrv::process(const cl::CommandQueue& queue, const cl::Image2D& inpImg, cl::Image2D& outImg, float minThresh, float maxThresh,
                           std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent)
{
    createIntImages(inpImg);
    gauss(queue, inpImg, *mGaussImg, events, pWaitEvent);
    gradient(queue, *mGaussImg, *mGradImg, events);
    suppress(queue, *mGradImg, *mNmesImg, events);
    binaryThreshold(queue, *mNmesImg, outImg, minThresh, maxThresh, events);
}

void CannyEdgePrv::process(const cl::CommandQueue& queue, const cl::ImageGL& inpImg, cl::ImageGL& outImg, float minThresh, float maxThresh,
                           std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent)
{
    createIntImages(inpImg);
    std::vector<cl::Memory> gl_objs = { inpImg, outImg };
    queue.enqueueAcquireGLObjects(&gl_objs);
    gauss(queue, inpImg, *mGaussImg, events, pWaitEvent);
    gradient(queue, *mGaussImg, *mGradImg, events);
    suppress(queue, *mGradImg, *mNmesImg, events);
    binaryThreshold(queue, *mNmesImg, outImg, minThresh, maxThresh, events);
    queue.enqueueReleaseGLObjects(&gl_objs);
}
