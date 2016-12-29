#include "OclHarrisCornerPrv.h"
#include "OclUtils.h"
#include <sstream>

using namespace Ocl;

HarrisCornerPrv::HarrisCornerPrv(cl::Context& ctxt, cl::CommandQueue& q)
    :mWidth(0),
     mHeight(0),
     mLocSizeX(16),
     mLocSizeY(16),
     mContext(ctxt),
     mQueue(q),
     mCoeffBuff(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 25),
     mCompact(ctxt, q)
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

HarrisCornerPrv::~HarrisCornerPrv()
{
}


void HarrisCornerPrv::init()
{
    cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
    mPgm = cl::Program(mContext, source);

    std::ostringstream options;
    options << " -DBLK_SIZE_X=" << mLocSizeX << " -DBLK_SIZE_Y=" << mLocSizeY;
    cl_int err = mPgm.build(options.str().c_str());

    mGradKernel = cl::Kernel(mPgm, "gradient");
    mEigenKernel = cl::Kernel(mPgm, "eigen");
    mCornerKernel = cl::Kernel(mPgm, "suppress_non_max");
}

void HarrisCornerPrv::loadGaussCoeffs()
{
    static float gaussCoeffs[] = {
        (float)(2.0 / 159.0), (float)(4.0 / 159.0), (float)(5.0 / 159.0), (float)(4.0 / 159.0), (float)(2.0 / 159.0),
        (float)(4.0 / 159.0), (float)(9.0 / 159.0), (float)(12.0 / 159.0), (float)(9.0 / 159.0), (float)(4.0 / 159.0),
        (float)(5.0 / 159.0), (float)(12.0 / 159.0), (float)(15.0 / 159.0), (float)(12.0 / 159.0), (float)(5.0 / 159.0),
        (float)(4.0 / 159.0), (float)(9.0 / 159.0), (float)(12.0 / 159.0), (float)(9.0 / 159.0), (float)(4.0 / 159.0),
        (float)(2.0 / 159.0), (float)(4.0 / 159.0), (float)(5.0 / 159.0), (float)(4.0 / 159.0), (float)(2.0 / 159.0) };

    float* pCoeff = mCoeffBuff.map(mQueue, CL_TRUE, CL_MAP_WRITE, 0, 25);
    memcpy(pCoeff, gaussCoeffs, 25*sizeof(float));
    mCoeffBuff.unmap(mQueue, pCoeff);
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

size_t HarrisCornerPrv::gradient(const cl::Image& inpImg, cl::Image& outImg)
{
    cl::Event event;
    mGradKernel.setArg(0, inpImg);
    mGradKernel.setArg(1, outImg);
    mQueue.enqueueNDRangeKernel(mGradKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), NULL, &event);
    event.wait();
    return kernelExecTime(mQueue, event);
}

size_t HarrisCornerPrv::eigen(const cl::Image& inpImg, cl::Image& outImg)
{
    cl::Event event;
    mEigenKernel.setArg(0, inpImg);
    mEigenKernel.setArg(1, outImg);
    mEigenKernel.setArg(2, mCoeffBuff.buffer());
    mQueue.enqueueNDRangeKernel(mEigenKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), NULL, &event);
    event.wait();
    return kernelExecTime(mQueue, event);
}

size_t HarrisCornerPrv::suppress(const cl::Image& inpImg, cl::Image& outImg, float value)
{
    cl::Event event;
    mCornerKernel.setArg(0, inpImg);
    mCornerKernel.setArg(1, outImg);
    mCornerKernel.setArg(2, value);
    mQueue.enqueueNDRangeKernel(mCornerKernel, cl::NullRange, cl::NDRange(mWidth, mHeight), cl::NDRange(mLocSizeX, mLocSizeY), NULL, &event);
    event.wait();
    return kernelExecTime(mQueue, event);
}

size_t HarrisCornerPrv::process(const cl::Image2D& inpImg, Ocl::DataBuffer<Ocl::Pos>& corners, float value, size_t& count)
{
    createIntImages(inpImg);
    size_t time = gradient(inpImg, *mIxIyImg);
    time += eigen(*mIxIyImg, *mEigenImg);
    time += suppress(*mEigenImg, *mCornerImg, value);
    time += mCompact.process(*mCornerImg, corners, 1.0f, count);
    return time;
}

size_t HarrisCornerPrv::process(const cl::ImageGL& inpImg, Ocl::DataBuffer<Ocl::Pos>& corners, float value, size_t& count)
{
    createIntImages(inpImg);
    std::vector<cl::Memory> gl_objs = { inpImg };
    mQueue.enqueueAcquireGLObjects(&gl_objs);
    size_t time = gradient(inpImg, *mIxIyImg);
    mQueue.enqueueReleaseGLObjects(&gl_objs);
    time += eigen(*mIxIyImg, *mEigenImg);
    time += suppress(*mEigenImg, *mCornerImg, value);
    time += mCompact.process(*mCornerImg, corners, 1.0f, count);
    return time;
}
