#pragma once

#include <CL/cl.hpp>
#include <memory>

class OclCompact
{
public:
	OclCompact(cl::Context& ctxt, cl::CommandQueue& queue);
	~OclCompact();

public:
	void process(cl::Buffer& inp, cl::Buffer& out, size_t& outCount);

private:
	size_t doScan(cl::Buffer& buffer);
	void createIntBuffer(size_t buffSize);

private:
	const int mScanBlkSize;
	const int mReduceBlkSize;
	cl::Context& mContext;
	cl::CommandQueue& mQueue;

	cl::Program mProgram;
	cl::Kernel mScanKernel;
	cl::Kernel mReduceKernel;
	cl::Kernel mAddResKernel;
	cl::Kernel mGatherScanKernel;
	cl::Kernel mCompactScanKernel;

	cl::Buffer mIntBuffScan;
	std::unique_ptr<cl::Buffer> mIntBuffReduce;

	static const char sSource[];
};
