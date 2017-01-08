#include "OclHarrisCornerPrv.h"
#include "OclUtils.h"
#include <sstream>

using namespace Ocl;

static const float gaussCoeffs[] =
{
    (float)(2.0/159.0), (float)(4.0/159.0), (float)(5.0/159.0), (float)(4.0/159.0), (float)(2.0/159.0),
    (float)(4.0/159.0), (float)(9.0/159.0), (float)(12.0/159.0), (float)(9.0/159.0), (float)(4.0/159.0),
    (float)(5.0/159.0), (float)(12.0/159.0), (float)(15.0/159.0), (float)(12.0/159.0), (float)(5.0/159.0),
    (float)(4.0/159.0), (float)(9.0/159.0), (float)(12.0/159.0), (float)(9.0/159.0), (float)(4.0/159.0),
    (float)(2.0/159.0), (float)(4.0/159.0), (float)(5.0/159.0), (float)(4.0/159.0), (float)(2.0/159.0)
};

HarrisCornerPrv::HarrisCornerPrv(const cl::Context& ctxt)
    :mWidth(0),
     mHeight(0),
     mIsLoaded(false),
     mLocSizeX(16),
     mLocSizeY(16),
     mContext(ctxt),
     mCoeffBuff(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 25),
     mCompact(ctxt)
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

    std::ostringstream options;
    options << " -DBLK_SIZE_X=" << mLocSizeX << " -DBLK_SIZE_Y=" << mLocSizeY;
    mPgm.build(options.str().c_str());

    mGradKernel = cl::Kernel(mPgm, "gradient");
    mEigenKernel = cl::Kernel(mPgm, "eigen");
    mCornerKernel = cl::Kernel(mPgm, "suppress_non_max");
}

void HarrisCornerPrv::loadGaussCoeffs(const cl::CommandQueue& queue)
{
    if (mIsLoaded == false)
    {
        mIsLoaded = true;
        float* pCoeff = mCoeffBuff.map(queue, CL_TRUE, CL_MAP_WRITE, 0, 25);
        memcpy(pCoeff, gaussCoeffs, 25 * sizeof(float));
        mCoeffBuff.unmap(queue, pCoeff);
    }
}

void HarrisCornerPrv::checkLocalGroupSizes()
{
    size_t xSize = Ocl::localGroupSize(mWidth);
    size_t ySize = Ocl::localGroupSize(mHeight);
    if (xSize != mLocSizeX || ySize != mLocSizeY)
    {
        mLocSizeX = xSize;
        mLocSizeY = ySize;
        init();
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
    checkLocalGroupSizes();
}

size_t HarrisCornerPrv::gradient(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg)
{
    cl::Event event;
    mGradKernel.setArg(0, inpImg);
    mGradKernel.setArg(1, outImg);
    queue.enqueueNDRangeKernel(mGradKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), NULL, &event);
    event.wait();
    return kernelExecTime(queue, event);
}

size_t HarrisCornerPrv::eigen(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg)
{
    cl::Event event;
    mEigenKernel.setArg(0, inpImg);
    mEigenKernel.setArg(1, outImg);
    mEigenKernel.setArg(2, mCoeffBuff.buffer());
    queue.enqueueNDRangeKernel(mEigenKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), NULL, &event);
    event.wait();
    return kernelExecTime(queue, event);
}

size_t HarrisCornerPrv::suppress(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, float value)
{
    cl::Event event;
    mCornerKernel.setArg(0, inpImg);
    mCornerKernel.setArg(1, outImg);
    mCornerKernel.setArg(2, value);
    queue.enqueueNDRangeKernel(mCornerKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), NULL, &event);
    event.wait();
    return kernelExecTime(queue, event);
}

size_t HarrisCornerPrv::process(const cl::CommandQueue& queue, const cl::Image2D& inpImg, Ocl::DataBuffer<Ocl::Pos>& corners, float value, size_t& count)
{
    loadGaussCoeffs(queue);
    createIntImages(inpImg);
    size_t time = gradient(queue, inpImg, *mIxIyImg);
    time += eigen(queue, *mIxIyImg, *mEigenImg);
    time += suppress(queue, *mEigenImg, *mCornerImg, value);
    time += mCompact.process(queue, *mCornerImg, corners, 1.0f, count);
    return time;
}

size_t HarrisCornerPrv::process(const cl::CommandQueue& queue, const cl::ImageGL& inpImg, Ocl::DataBuffer<Ocl::Pos>& corners, float value, size_t& count)
{
    loadGaussCoeffs(queue);
    createIntImages(inpImg);
    std::vector<cl::Memory> gl_objs = { inpImg };
    queue.enqueueAcquireGLObjects(&gl_objs);
    size_t time = gradient(queue, inpImg, *mIxIyImg);
    queue.enqueueReleaseGLObjects(&gl_objs);
    time += eigen(queue, *mIxIyImg, *mEigenImg);
    time += suppress(queue, *mEigenImg, *mCornerImg, value);
    time += mCompact.process(queue, *mCornerImg, corners, 1.0f, count);
    return time;
}
