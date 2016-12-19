#ifndef _OCL_HISTOGRAM_H_
#define _OCL_HISTOGRAM_H_

#include <CL/cl.hpp>
#include <memory>

class OclHistogram
{
public:
	OclHistogram(cl::Context& ctxt, cl::CommandQueue& queue);
	~OclHistogram();

public:
	void compute(const cl::Image2D& img, cl::Buffer& histBins);

private:
	cl::Context& mContext;
	cl::CommandQueue& mQueue;
	std::unique_ptr<cl::Buffer> mTempBuff;

	cl::Program mPgm;

	cl::Kernel mIntHistKernel;
	cl::Kernel mAccHistKernel;

	static const char sHistPgmSrc[];
};

#endif
