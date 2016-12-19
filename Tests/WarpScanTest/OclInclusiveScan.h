#pragma once

#include <CL/cl.hpp>

class OclInclusiveScan
{
public:
	OclInclusiveScan(int depth, cl::Context& ctxt, cl::CommandQueue& queue);
	~OclInclusiveScan();

public:
	void process(cl::Buffer& buffer);

private:
	const int mDepth;
	const int mBlkSize;
	cl::Context& mContext;
	cl::CommandQueue& mQueue;

	cl::Buffer mIntBuff;

	cl::Program mProgram;
	
	cl::Kernel mScanKernel;
	cl::Kernel mAddResKernel;
	cl::Kernel mGatherScanKernel;

	static const char sSource[];
};
