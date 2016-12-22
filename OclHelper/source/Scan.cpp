#include "Scan.h"

#include <sstream>

using namespace Ocl;

#define OCL_PROGRAM_SOURCE(s) #s

const char Scan::sSource[] = OCL_PROGRAM_SOURCE(
inline void warp_scan(int wid, int i, local volatile int* sh_data)
{
	if (wid >= 1) sh_data[i] += sh_data[i-1];
	if (wid >= 2) sh_data[i] += sh_data[i-2];
	if (wid >= 4) sh_data[i] += sh_data[i-4];
	if (wid >= 8) sh_data[i] += sh_data[i-8];
	if (wid >= 16) sh_data[i] += sh_data[i-16];
	/*#if (WARP_SIZE >= 64)
	if (wid >= 32) sh_data[i] += sh_data[i-32];
	#endif*/
}

inline void block_scan(local volatile int* sh_data)
{
	const int i = get_local_id(0);
	const int j = i%WARP_SIZE;
	const int k = i/WARP_SIZE;

	warp_scan(j, i, sh_data);
	warp_scan(j, (int)(get_local_size(0) + i), sh_data);
	barrier(CLK_LOCAL_MEM_FENCE);

	if (i == 0)
	{
		sh_data[SH_MEM_SIZE] = 0;
	}

	if (i < ((SH_MEM_SIZE/WARP_SIZE)-1))
	{
		sh_data[SH_MEM_SIZE+i+1] = sh_data[(i*WARP_SIZE)+(WARP_SIZE-1)];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (k == 0)
	{
		warp_scan(j, (SH_MEM_SIZE+i), sh_data);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int p = 0; p < 2; p++)
	{
		int n = (i*2)+p;
		sh_data[n] += sh_data[SH_MEM_SIZE+(n/WARP_SIZE)];
	}
}

kernel void prefix_sum(global int* p_data, int count)
{
	local int sh_data[SH_MEM_SIZE+WARP_SIZE];
	for (int i = 0; i < 2; i++)
	{
		int j = i+(2*get_local_id(0));
		int k = i+(2*get_global_id(0));
		sh_data[j] = (k < count)?p_data[k]:0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	block_scan(sh_data);
	for (int i = 0; i < 2; i++)
	{
		int j = i+(2*get_local_id(0));
		int k = i+(2*get_global_id(0));
		if (k < count)
		{
			p_data[k] = sh_data[j];
		}
	}
}

kernel void gather_scan(global const int* p_data, int start, int count, global int* p_out_data)
{
	local int sh_data[SH_MEM_SIZE+WARP_SIZE];
	const int i = (get_local_id(0)*2);
	const int j = (SH_MEM_SIZE*i)+(start-1); //mad(i, SH_MEM_SIZE, (start-1));
	sh_data[i] = (j < count) ? p_data[j] : 0;
	sh_data[i+1] = ((j+SH_MEM_SIZE) < count) ? p_data[j+SH_MEM_SIZE]:0;
	barrier(CLK_LOCAL_MEM_FENCE);
	block_scan(sh_data);
	barrier(CLK_LOCAL_MEM_FENCE);
	p_out_data[i] = sh_data[i];
	p_out_data[i+1] = sh_data[i+1];
}

kernel void add_data(global int* p_data, int offset, int out_count, global const int* p_in_data)
{
	local int sh_data;
	if (get_local_id(0) == 0)
	{
		sh_data = p_in_data[get_group_id(0)];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	const int i = (SH_MEM_SIZE*get_group_id(0))+(4*get_local_id(0))+offset;
	for (int j = i; j < (i+4); j++)
	{
		if (j < out_count)
		{
			p_data[j] += sh_data;
		}
	}
});

const int WARP_SIZE = 32;

Scan::Scan(cl::Context& ctxt, cl::CommandQueue& queue)
	:mDepth(8),
	 mBlkSize(1<<8),
	 mContext(ctxt),
	 mQueue(queue),
	 mIntBuff(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, (size_t)(mBlkSize*sizeof(int)))
{
	std::ostringstream options;
	options << "-DWARP_SIZE=" << WARP_SIZE << " -DSH_MEM_SIZE=" << (1 << mDepth);

	cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
	mProgram = cl::Program(mContext, source);
	mProgram.build(options.str().c_str());

	mScanKernel = cl::Kernel(mProgram, "prefix_sum");
	mAddResKernel = cl::Kernel(mProgram, "add_data");
	mGatherScanKernel = cl::Kernel(mProgram, "gather_scan");
}

Scan::~Scan()
{
}

size_t Scan::process(Ocl::DataBuffer<int>& buffer)
{
	cl::Event event;
	size_t buffSize = buffer.count();

	mScanKernel.setArg(0, buffer.buffer());
	mScanKernel.setArg(1, (int)buffSize);

	size_t gSize = (buffSize/mBlkSize);
	gSize += ((buffSize%mBlkSize) == 0) ? 0 : 1;
	gSize *= mBlkSize;
	mQueue.enqueueNDRangeKernel(mScanKernel, cl::NullRange, cl::NDRange(gSize/2), cl::NDRange(mBlkSize/2), NULL, &event);
	event.wait();
	size_t time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>()-event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

	for (size_t i = mBlkSize; i < buffSize; i += (mBlkSize*mBlkSize))
	{
		mGatherScanKernel.setArg(0, buffer);
		mGatherScanKernel.setArg(1, (int)i);
		mGatherScanKernel.setArg(2, (int)buffSize);
		mGatherScanKernel.setArg(3, mIntBuff);
		mQueue.enqueueNDRangeKernel(mGatherScanKernel, cl::NullRange, cl::NDRange(mBlkSize/2), cl::NDRange(mBlkSize/2), NULL, &event);
		event.wait();
		time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>()-event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

		mAddResKernel.setArg(0, buffer);
		mAddResKernel.setArg(1, (int)i);
		mAddResKernel.setArg(2, (int)buffSize);
		mAddResKernel.setArg(3, mIntBuff);
		mQueue.enqueueNDRangeKernel(mAddResKernel, cl::NullRange, cl::NDRange(mBlkSize*mBlkSize/4), cl::NDRange(mBlkSize/4), NULL, &event);
		event.wait();
		time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>()-event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
	}
	//printf("\nKernel Time: %llf us", ((double)time)/1000.0);
	return time;
}
