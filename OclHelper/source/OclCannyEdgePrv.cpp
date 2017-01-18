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
    mNmesKernel = cl::Kernel(mPgm, "non_max_edge_suppress");
    mBinThreshKernel = cl::Kernel(mPgm, "binary_threshold");
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

size_t CannyEdgePrv::gradient(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg)
{
    cl::Event event;
    mGradKernel.setArg(0, inpImg);
    mGradKernel.setArg(1, outImg);
    queue.enqueueNDRangeKernel(mGradKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), NULL, &event);
    event.wait();
    return kernelExecTime(queue, event);
}

size_t CannyEdgePrv::gauss(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg)
{
    cl::Event event;
    mGaussKernel.setArg(0, inpImg);
    mGaussKernel.setArg(1, outImg);
    queue.enqueueNDRangeKernel(mGaussKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), NULL, &event);
    event.wait();
    return kernelExecTime(queue, event);
}

size_t CannyEdgePrv::suppress(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg)
{
    cl::Event event;
    mNmesKernel.setArg(0, inpImg);
    mNmesKernel.setArg(1, outImg);
    queue.enqueueNDRangeKernel(mNmesKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), NULL, &event);
    event.wait();
    return kernelExecTime(queue, event);
}

void CannyEdgePrv::checkLocalGroupSizes()
{
    mLocSizeX = Ocl::localGroupSize(mWidth);
    mLocSizeY = Ocl::localGroupSize(mHeight);
}

size_t CannyEdgePrv::binaryThreshold(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, float minThresh, float maxThresh)
{
    cl::Event event;
    mBinThreshKernel.setArg(0, inpImg);
    mBinThreshKernel.setArg(1, outImg);
    mBinThreshKernel.setArg(2, minThresh);
    mBinThreshKernel.setArg(3, maxThresh);
    queue.enqueueNDRangeKernel(mBinThreshKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), NULL, &event);
    event.wait();
    return kernelExecTime(queue, event);
}

size_t CannyEdgePrv::process(const cl::CommandQueue& queue, const cl::Image2D& inpImg, cl::Image2D& outImg, float minThresh, float maxThresh)
{
    createIntImages(inpImg);
    size_t time = gauss(queue, inpImg, *mGaussImg);
    time += gradient(queue, *mGaussImg, *mGradImg);
    time += suppress(queue, *mGradImg, *mNmesImg);
    time += binaryThreshold(queue, *mNmesImg, outImg, minThresh, maxThresh);
    return time;
}

size_t CannyEdgePrv::process(const cl::CommandQueue& queue, const cl::ImageGL& inpImg, cl::ImageGL& outImg, float minThresh, float maxThresh)
{
    createIntImages(inpImg);
    std::vector<cl::Memory> gl_objs = { inpImg, outImg };
    queue.enqueueAcquireGLObjects(&gl_objs);
    size_t time = gauss(queue, inpImg, *mGaussImg);
    time += gradient(queue, *mGaussImg, *mGradImg);
    time += suppress(queue, *mGradImg, *mNmesImg);
    time += binaryThreshold(queue, *mNmesImg, outImg, minThresh, maxThresh);
    queue.enqueueReleaseGLObjects(&gl_objs);
    return time;
}
