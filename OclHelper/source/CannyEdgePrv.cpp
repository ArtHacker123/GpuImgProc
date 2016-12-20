#include "CannyEdgePrv.h"
#include <sstream>

using namespace Ocl;

CannyEdgePrv::CannyEdgePrv(cl::Context& ctxt, cl::CommandQueue& q)
	:mContext(ctxt),
	 mQueue(q),
	 mCoeffBuff(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 25)
{
	cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
	mPgm = cl::Program(mContext, source);
	cl_int err = mPgm.build();

	mGradKernel = cl::Kernel(mPgm, "gradient");
	mEigenKernel = cl::Kernel(mPgm, "eigen");
	mCornerKernel = cl::Kernel(mPgm, "suppress_non_max");

	static float gaussCoeffs[] = { 
		(float)(2.0/159.0), (float)(4.0/159.0), (float)(5.0/159.0), (float)(4.0/159.0), (float)(2.0/159.0),
		(float)(4.0/159.0), (float)(9.0/159.0), (float)(12.0/159.0), (float)(9.0/159.0), (float)(4.0/159.0),
		(float)(5.0/159.0), (float)(12.0/159.0), (float)(15.0/159.0), (float)(12.0/159.0), (float)(5.0/159.0),
		(float)(4.0/159.0), (float)(9.0/159.0), (float)(12.0/159.0), (float)(9.0/159.0), (float)(4.0/159.0),
		(float)(2.0/159.0), (float)(4.0/159.0), (float)(5.0/159.0), (float)(4.0/159.0), (float)(2.0/159.0) };

    float* pCoeff = mCoeffBuff.map(mQueue, CL_TRUE, CL_MAP_WRITE, 0, 25);
	memcpy(pCoeff, gaussCoeffs, 25*sizeof(float));
    mCoeffBuff.unmap(mQueue, pCoeff);
}

CannyEdgePrv::~CannyEdgePrv()
{
}

void CannyEdgePrv::createIntImages(size_t w, size_t h)
{
	size_t width = 0, height = 0;
	if (mIxIyImg.get() != 0)
	{
		mIxIyImg->getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
		mIxIyImg->getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);
	}

	if (w != width || h != height)
	{
		mIxIyImg.reset(new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), w, h));
		mEigenImg.reset(new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), w, h));
		mCornerImg.reset(new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_HALF_FLOAT), w, h));
	}
}

size_t CannyEdgePrv::gradient(const cl::Image& inpImg, cl::Image& outImg)
{
	cl::Event event;
	size_t width, height;
	inpImg.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
	inpImg.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);

	mGradKernel.setArg(0, inpImg);
	mGradKernel.setArg(1, outImg);
	mQueue.enqueueNDRangeKernel(mGradKernel, cl::NullRange, cl::NDRange(width, height), cl::NDRange(8, 8), NULL, &event);
	event.wait();
	return (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
}

size_t CannyEdgePrv::eigen(const cl::Image& inpImg, cl::Image& outImg)
{
	cl::Event event;
	size_t width, height;
	inpImg.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
	inpImg.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);

	mEigenKernel.setArg(0, inpImg);
	mEigenKernel.setArg(1, outImg);
	mEigenKernel.setArg(2, mCoeffBuff);
	mQueue.enqueueNDRangeKernel(mEigenKernel, cl::NullRange, cl::NDRange(width, height), cl::NDRange(16, 16), NULL, &event);
	event.wait();
	return (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
}

size_t CannyEdgePrv::suppress(const cl::Image& inpImg, cl::Image& outImg, float value)
{
	cl::Event event;
	size_t width, height;
	inpImg.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
	inpImg.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);

	mCornerKernel.setArg(0, inpImg);
	mCornerKernel.setArg(1, outImg);
	mCornerKernel.setArg(2, value);
	mQueue.enqueueNDRangeKernel(mCornerKernel, cl::NullRange, cl::NDRange(width, height), cl::NDRange(16, 16), NULL, &event);
	event.wait();
	return (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
}

size_t CannyEdgePrv::process(const cl::Image2D& inpImg, cl::Image2D& outImage, float minThresh, float maxThresh)
{
	size_t width, height;
	inpImg.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
	inpImg.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);
	createIntImages(width, height);
	size_t time = gradient(inpImg, *mIxIyImg);
	time += eigen(*mIxIyImg, *mEigenImg);
	//time += suppress(*mEigenImg, *mCornerImg, value);
	//time += mCompact.process(*mCornerImg, corners, 1.0f, count);
	//printf("\nKernel Time: %lf ms", ((double)time)/1000000.0);
    return time;
}

size_t CannyEdgePrv::process(const cl::ImageGL& inpImg, cl::ImageGL& outImg, float minThresh, float maxThresh)
{
	size_t width, height;
	inpImg.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
	inpImg.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);
	createIntImages(width, height);
    std::vector<cl::Memory> gl_objs = { inpImg, outImg };
    mQueue.enqueueAcquireGLObjects(&gl_objs);
	size_t time = gradient(inpImg, outImg);
    mQueue.enqueueReleaseGLObjects(&gl_objs);
	//time += eigen(*mIxIyImg, *mEigenImg);
	//time += suppress(*mEigenImg, *mCornerImg, value);
	//time += mCompact.process(*mCornerImg, corners, 1.0f, count);
	//printf("\nKernel Time: %lf ms", ((double)time)/1000000.0);
    return time;
}
