#include "OclCompact.h"

#include <sstream>

#define OCL_PROGRAM_SOURCE(s) #s

const char OclCompact::sSource[] = OCL_PROGRAM_SOURCE(
inline void block_reduce_sum(local int* sh_data)
{
	if (get_local_id(0) < 32) sh_data[get_local_id(0)] += sh_data[32+get_local_id(0)];
	if (get_local_id(0) < 16) sh_data[get_local_id(0)] += sh_data[16+get_local_id(0)];
    if (get_local_id(0) < 8) sh_data[get_local_id(0)] += sh_data[8+get_local_id(0)];
    if (get_local_id(0) < 4) sh_data[get_local_id(0)] += sh_data[4+get_local_id(0)];
    if (get_local_id(0) < 2) sh_data[get_local_id(0)] += sh_data[2+get_local_id(0)];
    if (get_local_id(0) < 1) sh_data[get_local_id(0)] += sh_data[1+get_local_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void reduce_sum(global const int* p_data, int count, global int* p_block_sum)
{
	local int sh_data[SH_MEM_SIZE_REDUCE];
	int index = (4*SH_MEM_SIZE_REDUCE*get_group_id(0)) + get_local_id(0);
	sh_data[get_local_id(0)] = (index < count)?p_data[index]:0;
	index += get_local_size(0);
	sh_data[get_local_id(0)] += (index < count)?p_data[index]:0;
	index += get_local_size(0);
	sh_data[get_local_id(0)] += (index < count)?p_data[index]:0;
	index += get_local_size(0);
	sh_data[get_local_id(0)] += (index < count)?p_data[index]:0;
	barrier(CLK_LOCAL_MEM_FENCE);
	block_reduce_sum(sh_data);
	if (get_local_id(0) == 0)
	{
		p_block_sum[get_group_id(0)] = sh_data[0];
	}
}

inline void warp_scan(int wid, int i, local int* sh_data)
{
	if (wid >= 1) sh_data[i] += sh_data[i - 1];
	if (wid >= 2) sh_data[i] += sh_data[i - 2];
	if (wid >= 4) sh_data[i] += sh_data[i - 4];
	if (wid >= 8) sh_data[i] += sh_data[i - 8];
	if (wid >= 16) sh_data[i] += sh_data[i-16];
	/*#if (WARP_SIZE >= 64)
	if (wid >= 32) sh_data[i] += sh_data[i-32];
	#endif*/
}

inline void block_scan(local int* sh_data)
{
	const int i = get_local_id(0);
	const int j = i%WARP_SIZE;
	const int k = i/WARP_SIZE;

	for (int p = 0; p < 2; p++)
	{
		int index = (get_local_size(0)*p) + i;
		warp_scan(j, index, sh_data);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (k == 0)
	{
		if (i == 0)
		{
			sh_data[SH_MEM_SIZE] = 0;
		}
		else
		{
			sh_data[SH_MEM_SIZE + i] = sh_data[((i - 1)*WARP_SIZE) + (WARP_SIZE - 1)];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (k == 0)
	{
		warp_scan(j, (SH_MEM_SIZE + i), sh_data);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int p = 0; p < 2; p++)
	{
		int n = (i * 2) + p;
		sh_data[n] += sh_data[SH_MEM_SIZE + (n / WARP_SIZE)];
	}
}

kernel void prefix_sum(global int2* p_data, int count)
{
	local int2 sh_data[(SH_MEM_SIZE+WARP_SIZE)/2];
	sh_data[get_local_id(0)] = p_data[get_global_id(0)];
	/*local int sh_data[SH_MEM_SIZE+WARP_SIZE];
	for (int i = 0; i < 2; i++)
	{
		int j = i + (2 * get_local_id(0));
		int k = i + (2 * get_global_id(0));
		sh_data[j] = (k < count) ? p_data[k] : 0;
	}*/
	barrier(CLK_LOCAL_MEM_FENCE);
	block_scan(sh_data);
	/*for (int i = 0; i < 2; i++)
	{
		int j = i + (2 * get_local_id(0));
		int k = i + (2 * get_global_id(0));
		if (k < count)
		{
			p_data[k] = sh_data[j];
		}
	}*/
	p_data[get_global_id(0)] = sh_data[get_local_id(0)];
}

kernel void gather_scan(global const int* p_data, int start, int count, global int* p_out_data)
{
	local int sh_data[SH_MEM_SIZE + WARP_SIZE];
	const int i = (get_local_id(0) * 2);
	const int j = mad(i, SH_MEM_SIZE, (start - 1));
	sh_data[i] = (j < count) ? p_data[j] : 0;
	sh_data[i + 1] = ((j + SH_MEM_SIZE) < count) ? p_data[j + SH_MEM_SIZE] : 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	block_scan(sh_data);
	barrier(CLK_LOCAL_MEM_FENCE);
	p_out_data[i] = sh_data[i];
	p_out_data[i + 1] = sh_data[i + 1];
}

kernel void add_data(global int* p_data, int offset, int out_count, global const int* p_in_data)
{
	local int sh_data;
	if (get_local_id(0) == 0)
	{
		sh_data = p_in_data[get_group_id(0)];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	const int i = (SH_MEM_SIZE*get_group_id(0)) + (4 * get_local_id(0)) + offset;
	for (int j = i; j < (i + 4); j++)
	{
		if (j < out_count)
		{
			p_data[j] += sh_data;
		}
	}
}

kernel void prefix_sum_compact(global int* p_data, int count, global int* p_blk_sum, global int* p_out, int maxOutSize, global int* p_out_size)
{
	local int sum_blk;
	local int sh_data[SH_MEM_SIZE+WARP_SIZE];
	for (int i = 0; i < 2; i++)
	{
		int j = i + (2 * get_local_id(0));
		int k = i + (2 * get_global_id(0));
		sh_data[j] = (k < count) ? p_data[k] : 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	block_scan(sh_data);

	if (get_local_id(0) == 0)
	{
		sum_blk = (get_group_id(0) == 0)?0:p_blk_sum[get_group_id(0)-1];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (get_global_id(0) == (get_global_size(0)-1))
	{
		p_out_size[0] = sh_data[SH_MEM_SIZE-1] + sum_blk;
	}

	for (int i = 0; i < 2; i++)
	{
		int j = i + (2 * get_local_id(0));
		int k = i + (2 * get_global_id(0));
		if (k < count && p_data[k] == 1)
		{
			sh_data[j] += sum_blk;
			if (sh_data[j] < maxOutSize) p_out[sh_data[j]-1] = k;
		}
	}
}
);

OclCompact::OclCompact(cl::Context& ctxt, cl::CommandQueue& queue)
	:mScanBlkSize(256),
	 mReduceBlkSize(64),
	 mContext(ctxt),
	 mQueue(queue),
	 mIntBuffScan(mContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, (size_t)(mScanBlkSize*sizeof(int)))
{
	std::ostringstream options;
	options << "-DSH_MEM_SIZE_REDUCE=" << 64 << " -DSH_MEM_SIZE=" << 256 << " -DWARP_SIZE=" << 32;

	cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
	mProgram = cl::Program(mContext, source);
	mProgram.build(options.str().c_str());

	mScanKernel = cl::Kernel(mProgram, "prefix_sum");
	mReduceKernel = cl::Kernel(mProgram, "reduce_sum");
	mAddResKernel = cl::Kernel(mProgram, "add_data");
	mGatherScanKernel = cl::Kernel(mProgram, "gather_scan");
	mCompactScanKernel = cl::Kernel(mProgram, "prefix_sum_compact");
}

OclCompact::~OclCompact()
{
}

void OclCompact::createIntBuffer(size_t buffSize)
{
	size_t memSize = 0;
	size_t intBuffSize = (buffSize/(4*mReduceBlkSize))+(((buffSize%(4*mReduceBlkSize))==0)?0:1);
	if (mIntBuffReduce.get() != 0)
	{
		mIntBuffReduce->getInfo<size_t>(CL_MEM_SIZE, &memSize);
		memSize /= sizeof(int);
	}

	if (memSize < intBuffSize)
	{
		mIntBuffReduce.reset(new cl::Buffer(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, (size_t)(intBuffSize*sizeof(int))));
	}
}

void OclCompact::process(cl::Buffer& inp, cl::Buffer& out, size_t& outCount)
{
	cl::Event event;
	size_t buffSize = 0;
	size_t maxOutSize = 0;
	cl::Buffer outSize(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, (size_t)sizeof(int));

	inp.getInfo<size_t>(CL_MEM_SIZE, &buffSize);
	buffSize = buffSize/sizeof(int);

	out.getInfo<size_t>(CL_MEM_SIZE, &maxOutSize);
	maxOutSize = maxOutSize/sizeof(int);

	createIntBuffer(buffSize);

	mReduceKernel.setArg(0, inp);
	mReduceKernel.setArg(1, (int)buffSize);
	mReduceKernel.setArg(2, *mIntBuffReduce);

	size_t groupCount = (buffSize/4)+((buffSize % 4) == 0 ? 0 : 1);
	mQueue.enqueueNDRangeKernel(mReduceKernel, cl::NullRange, cl::NDRange(groupCount), cl::NDRange(mReduceBlkSize), NULL, &event);
	event.wait();
	size_t time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>()-event.getProfilingInfo<CL_PROFILING_COMMAND_START>();	

	time += doScan(*mIntBuffReduce);

	mCompactScanKernel.setArg(0, inp);
	mCompactScanKernel.setArg(1, (int)buffSize);
	mCompactScanKernel.setArg(2, *mIntBuffReduce);
	mCompactScanKernel.setArg(3, out);
	mCompactScanKernel.setArg(4, (int)maxOutSize);
	mCompactScanKernel.setArg(5, outSize);

	size_t gSize = ((buffSize/mScanBlkSize) + (((buffSize%mScanBlkSize) == 0) ? 0 : 1))*mScanBlkSize;
	mQueue.enqueueNDRangeKernel(mCompactScanKernel, cl::NullRange, cl::NDRange(gSize/2), cl::NDRange(mScanBlkSize/2), NULL, &event);
	event.wait();
	time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

	int* pData = (int *)mQueue.enqueueMapBuffer(outSize, CL_TRUE, CL_MAP_READ, 0, sizeof(int));
	outCount = *pData;
	mQueue.enqueueUnmapMemObject(outSize, pData);

	printf("\nKernel Time: %llf us", ((double)time)/1000.0);
}

size_t OclCompact::doScan(cl::Buffer& buffer)
{
	cl::Event event;
	size_t buffSize = 0;

	buffer.getInfo<size_t>(CL_MEM_SIZE, &buffSize);
	buffSize = buffSize / sizeof(int);

	mScanKernel.setArg(0, buffer);
	mScanKernel.setArg(1, (int)buffSize);

	size_t gSize = ((buffSize/mScanBlkSize) + (((buffSize%mScanBlkSize) == 0) ? 0 : 1))*mScanBlkSize;
	mQueue.enqueueNDRangeKernel(mScanKernel, cl::NullRange, cl::NDRange(gSize/2), cl::NDRange(mScanBlkSize/2), NULL, &event);
	event.wait();
	size_t time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>()-event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

	for (size_t i = mScanBlkSize; i < buffSize; i += (mScanBlkSize*mScanBlkSize))
	{
		mGatherScanKernel.setArg(0, buffer);
		mGatherScanKernel.setArg(1, (int)i);
		mGatherScanKernel.setArg(2, (int)buffSize);
		mGatherScanKernel.setArg(3, mIntBuffScan);
		mQueue.enqueueNDRangeKernel(mGatherScanKernel, cl::NullRange, cl::NDRange(mScanBlkSize / 2), cl::NDRange(mScanBlkSize/2), NULL, &event);
		event.wait();
		time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>()-event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

		mAddResKernel.setArg(0, buffer);
		mAddResKernel.setArg(1, (int)i);
		mAddResKernel.setArg(2, (int)buffSize);
		mAddResKernel.setArg(3, mIntBuffScan);
		mQueue.enqueueNDRangeKernel(mAddResKernel, cl::NullRange, cl::NDRange(mScanBlkSize*mScanBlkSize/4), cl::NDRange(mScanBlkSize/4), NULL, &event);
		event.wait();
		time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>()-event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
	}
	//printf("\nKernel Time: %llf us", ((double)time) / 1000.0);

	return time;
}
