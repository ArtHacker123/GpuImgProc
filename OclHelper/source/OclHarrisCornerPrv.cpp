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
    
    std::vector<cl::Device> devices = mContext.getInfo<CL_CONTEXT_DEVICES>();
    std::string version;
    devices[0].getInfo<std::string>(CL_DEVICE_OPENCL_C_VERSION, &version);

    std::ostringstream options;
    options << " -DBLK_SIZE_X=" << mLocSizeX << " -DBLK_SIZE_Y=" << mLocSizeY;
    mPgm.build(options.str().c_str());

    mGradKernel = cl::Kernel(mPgm, "gradient");
    mEigenKernel = cl::Kernel(mPgm, "eigen");
    mCornerKernel = cl::Kernel(mPgm, "nms");
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

size_t HarrisCornerPrv::process(const cl::CommandQueue& queue, const cl::Image2D& inpImg, Ocl::DataBuffer<cl_int2>& corners, float value, size_t& count)
{
    createIntImages(inpImg);
    size_t time = gradient(queue, inpImg, *mIxIyImg);
    time += eigen(queue, *mIxIyImg, *mEigenImg);
    time += suppress(queue, *mEigenImg, *mCornerImg, value);
    time += mCompact.process(queue, *mCornerImg, corners, 1.0f, count);
    return time;
}

size_t HarrisCornerPrv::process(const cl::CommandQueue& queue, const cl::ImageGL& inpImg, Ocl::DataBuffer<cl_int2>& corners, float value, size_t& count)
{
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
