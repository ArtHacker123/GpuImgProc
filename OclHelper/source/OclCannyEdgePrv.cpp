#include "OclCannyEdgePrv.h"
#include "OclUtils.h"
#include <sstream>

using namespace Ocl;

CannyEdgePrv::CannyEdgePrv(cl::Context& ctxt, cl::CommandQueue& q)
    :mWidth(0),
     mHeight(0),
     mLocSizeX(16),
     mLocSizeY(16),
     mContext(ctxt),
     mQueue(q),
     mCoeffBuff(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 25)
{
    try
    {
        init();
        loadGaussCoeffs();
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

void CannyEdgePrv::loadGaussCoeffs()
{
    static float gaussCoeffs[] = {
        (float)(2.0 / 159.0), (float)(4.0 / 159.0), (float)(5.0 / 159.0), (float)(4.0 / 159.0), (float)(2.0 / 159.0),
        (float)(4.0 / 159.0), (float)(9.0 / 159.0), (float)(12.0 / 159.0), (float)(9.0 / 159.0), (float)(4.0 / 159.0),
        (float)(5.0 / 159.0), (float)(12.0 / 159.0), (float)(15.0 / 159.0), (float)(12.0 / 159.0), (float)(5.0 / 159.0),
        (float)(4.0 / 159.0), (float)(9.0 / 159.0), (float)(12.0 / 159.0), (float)(9.0 / 159.0), (float)(4.0 / 159.0),
        (float)(2.0 / 159.0), (float)(4.0 / 159.0), (float)(5.0 / 159.0), (float)(4.0 / 159.0), (float)(2.0 / 159.0) };

    float* pCoeff = mCoeffBuff.map(mQueue, CL_TRUE, CL_MAP_WRITE, 0, 25);
    memcpy(pCoeff, gaussCoeffs, 25 * sizeof(float));
    mCoeffBuff.unmap(mQueue, pCoeff);
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

    checkLocalGroupSizes();
}

size_t CannyEdgePrv::gradient(const cl::Image& inpImg, cl::Image& outImg)
{
    cl::Event event;
    size_t shMemSize = (mLocSizeX+2)*(mLocSizeY+2)*sizeof(float);
    mGradKernel.setArg(0, inpImg);
    mGradKernel.setArg(1, outImg);
    mGradKernel.setArg(2, shMemSize, 0);
    mQueue.enqueueNDRangeKernel(mGradKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), NULL, &event);
    event.wait();
    return kernelExecTime(mQueue, event);
}

size_t CannyEdgePrv::gauss(const cl::Image& inpImg, cl::Image& outImg)
{
    cl::Event event;
    size_t shMemSize = (mLocSizeX+4)*(mLocSizeY+4)*sizeof(float);
    mGaussKernel.setArg(0, inpImg);
    mGaussKernel.setArg(1, outImg);
    mGaussKernel.setArg(2, mCoeffBuff);
    mGaussKernel.setArg(3, shMemSize, 0);
    mQueue.enqueueNDRangeKernel(mGaussKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), NULL, &event);
    event.wait();
    return kernelExecTime(mQueue, event);
}

size_t CannyEdgePrv::suppress(const cl::Image& inpImg, cl::Image& outImg)
{
    cl::Event event;
    size_t shMemSize = 2*(mLocSizeX+4)*(mLocSizeY+4)*sizeof(float);
    mNmesKernel.setArg(0, inpImg);
    mNmesKernel.setArg(1, outImg);
    mNmesKernel.setArg(2, shMemSize, 0);
    mQueue.enqueueNDRangeKernel(mNmesKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), NULL, &event);
    event.wait();
    return kernelExecTime(mQueue, event);
}

void CannyEdgePrv::checkLocalGroupSizes()
{
    mLocSizeX = Ocl::localGroupSize(mWidth);
    mLocSizeY = Ocl::localGroupSize(mHeight);
}

size_t CannyEdgePrv::binaryThreshold(const cl::Image& inpImg, cl::Image& outImg, float minThresh, float maxThresh)
{
    cl::Event event;
    size_t shMemSize = (mLocSizeX+2)*(mLocSizeY+2)*sizeof(float);
    mBinThreshKernel.setArg(0, inpImg);
    mBinThreshKernel.setArg(1, outImg);
    mBinThreshKernel.setArg(2, minThresh);
    mBinThreshKernel.setArg(3, maxThresh);
    mBinThreshKernel.setArg(4, shMemSize, 0);
    mQueue.enqueueNDRangeKernel(mBinThreshKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), NULL, &event);
    event.wait();
    return kernelExecTime(mQueue, event);
}

size_t CannyEdgePrv::process(const cl::Image2D& inpImg, cl::Image2D& outImg, float minThresh, float maxThresh)
{
    createIntImages(inpImg);
    size_t time = gauss(inpImg, *mGaussImg);
    time += gradient(*mGaussImg, *mGradImg);
    time += suppress(*mGradImg, *mNmesImg);
    time += binaryThreshold(*mNmesImg, outImg, minThresh, maxThresh);
    return time;
}

size_t CannyEdgePrv::process(const cl::ImageGL& inpImg, cl::ImageGL& outImg, float minThresh, float maxThresh)
{
    createIntImages(inpImg);
    std::vector<cl::Memory> gl_objs = { inpImg, outImg };
    mQueue.enqueueAcquireGLObjects(&gl_objs);
    size_t time = gauss(inpImg, *mGaussImg);
    time += gradient(*mGaussImg, *mGradImg);
    time += suppress(*mGradImg, *mNmesImg);
    time += binaryThreshold(*mNmesImg, outImg, minThresh, maxThresh);
    mQueue.enqueueReleaseGLObjects(&gl_objs);
    return time;
}
