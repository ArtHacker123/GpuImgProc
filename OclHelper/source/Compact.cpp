#include "Compact.h"

#include <sstream>

#define OCL_PROGRAM_SOURCE(s) #s

using namespace Ocl;

const char Compact::sSource[] = OCL_PROGRAM_SOURCE(
inline void block_reduce_sum(const int index, local volatile int* sh_data)
{
	if (index < 32) sh_data[index] += sh_data[32+index];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (index < 16) sh_data[index] += sh_data[16+index];
    if (index < 8) sh_data[index] += sh_data[8+index];
    if (index < 4) sh_data[index] += sh_data[4+index];
    if (index < 2) sh_data[index] += sh_data[2+index];
    if (index < 1) sh_data[index] += sh_data[1+index];
	barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void reduce_sum(read_only image2d_t image, float value, global int* p_block_sum)
{
	local int sh_data[SH_MEM_SIZE_REDUCE];
	const int x = 2*get_global_id(0);
	const int y = 2*get_global_id(1);
	const int index = (get_local_size(0)*get_local_id(1))+get_local_id(0);
	int data = (read_imagef(image, (int2)(x, y)).x >= value)?1:0;
	data += (read_imagef(image, (int2)(x+1, y)).x >= value)?1:0;
	data += (read_imagef(image, (int2)(x, y+1)).x >= value)?1:0;
	data += (read_imagef(image, (int2)(x+1, y+1)).x >= value)?1:0;
	sh_data[index] = data;
	barrier(CLK_LOCAL_MEM_FENCE);
	block_reduce_sum(index, sh_data);
	if (index == 0)
	{
		const int xwidth = get_image_width(image)/16;
		const int ywidth = get_image_height(image)/16;
		const int id = (get_group_id(1)*xwidth)+get_group_id(0);
		p_block_sum[id] = sh_data[0];
	}
}

inline void warp_scan(int wid, int i, local volatile int* sh_data)
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

inline void block_scan(const int i, local volatile int* sh_data)
{
	const int j = i%WARP_SIZE;
	const int k = i/WARP_SIZE;

	for (int p = 0; p < 2; p++)
	{
		int index = ((SH_MEM_SIZE/2)*p) + i;
		warp_scan(j, index, sh_data);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (i == 0)
	{
		sh_data[SH_MEM_SIZE] = 0;
	}

	if (i < 8)
	{
		sh_data[SH_MEM_SIZE + i + 1] = sh_data[(i*WARP_SIZE) + (WARP_SIZE - 1)];
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

kernel void prefix_sum_compact(read_only image2d_t image, float value, global int* p_blk_sum, global int2* p_out, int maxOutSize, global int* p_out_size)
{
	local int sum_blk;
	local int sh_data[SH_MEM_SIZE+WARP_SIZE];
	const int x = get_global_id(0);
	const int y = 2*get_global_id(1);
	const int index = (2*get_local_id(1)*get_local_size(0))+get_local_id(0);
	sh_data[index] = (read_imagef(image, (int2)(x, y)).x >= value)?1:0;
	sh_data[index+get_local_size(0)] = (read_imagef(image, (int2)(x, y+1)).x >= value)?1:0;
	barrier(CLK_LOCAL_MEM_FENCE);
	block_scan((get_local_id(1)*get_local_size(0))+get_local_id(0), sh_data);

	if (index == 0)
	{
		const int xwidth = get_image_width(image)/16;
		const int ywidth = get_image_height(image)/16;
		const int id = (get_group_id(1)*xwidth) + get_group_id(0);
		sum_blk = (id == 0) ? 0 : p_blk_sum[id-1];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (get_global_id(0) == (get_global_size(0) - 1))
	{
		int count = sh_data[SH_MEM_SIZE-1] + sum_blk;
		p_out_size[0] = (maxOutSize > count)?count:maxOutSize;
	}

	if (read_imagef(image, (int2)(x, y)).x >= value)
	{
		sh_data[index] += sum_blk;
		if (sh_data[index] < maxOutSize)
		{
			p_out[sh_data[index]-1] = (int2)(x, y);
		}
	}

	if (read_imagef(image, (int2)(x, y+1)).x >= value)
	{
		sh_data[index+get_local_size(0)] += sum_blk;
		if (sh_data[index+get_local_size(0)] < maxOutSize)
		{
			p_out[sh_data[index+get_local_size(0)]-1] = (int2)(x, y+1);
		}
	}
}

kernel void do_voting(read_only image2d_t inpImg, const float value, global int* p_data)
{
	const int width = get_image_width(inpImg);
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	p_data[(width*y)+x] = (read_imagef(inpImg, (int2)(x, y)).x >= value)?1:0;
}
);

Compact::Compact(cl::Context& ctxt, cl::CommandQueue& queue)
	:mContext(ctxt),
	 mQueue(queue),
	 mScanBlkSize(256),
	 mReduceBlkSize(64),
	 mIntBuffScan(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, (size_t)(mScanBlkSize*sizeof(int))),
	 mScan(ctxt, queue)
{
	std::ostringstream options;
	options << "-DSH_MEM_SIZE_REDUCE=" << 64 << " -DSH_MEM_SIZE=" << 256 << " -DWARP_SIZE=" << 32;

	cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
	mProgram = cl::Program(mContext, source);
	mProgram.build(options.str().c_str());

	mReduceKernel = cl::Kernel(mProgram, "reduce_sum");
	mCompactScanKernel = cl::Kernel(mProgram, "prefix_sum_compact");
}

Compact::~Compact()
{
}

void Compact::createIntBuffer(size_t buffSize)
{
	size_t memSize = 0;
	size_t intBuffSize = (buffSize/(4*mReduceBlkSize))+(((buffSize%(4*mReduceBlkSize))==0)?0:1);
	if (mBuffReduce.get() != 0)
	{
		memSize = mBuffReduce->count();
	}

	if (memSize < intBuffSize)
	{
		mBuffReduce.reset(new DataBuffer<int>(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, intBuffSize));
	}
}

size_t Compact::process(cl::Image& inpImage, Ocl::DataBuffer<Ocl::Pos>& out, float value, size_t& outCount)
{
	size_t width, height;
	cl::Buffer outSize(mContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, (size_t)sizeof(int));

	size_t maxOutSize = out.count();
	inpImage.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
	inpImage.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);

	createIntBuffer(width*height);

	mReduceKernel.setArg(0, inpImage);
	mReduceKernel.setArg(1, value);
	mReduceKernel.setArg(2, *mBuffReduce);
	cl::Event event;
	mQueue.enqueueNDRangeKernel(mReduceKernel, cl::NullRange, cl::NDRange(width/2, height/2), cl::NDRange(8, 8), NULL, &event);
	event.wait();
	size_t time = (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

	time += mScan.process(*mBuffReduce);

	mCompactScanKernel.setArg(0, inpImage);
	mCompactScanKernel.setArg(1, 0.5f);
	mCompactScanKernel.setArg(2, *mBuffReduce);
	mCompactScanKernel.setArg(3, out.buffer());
	mCompactScanKernel.setArg(4, (int)maxOutSize);
	mCompactScanKernel.setArg(5, outSize);

	mQueue.enqueueNDRangeKernel(mCompactScanKernel, cl::NullRange, cl::NDRange(width, height/2), cl::NDRange(16, 8), NULL, &event);
	event.wait();
	time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

	int* pData = (int *)mQueue.enqueueMapBuffer(outSize, CL_TRUE, CL_MAP_READ, 0, sizeof(int));
	outCount = *pData;
	mQueue.enqueueUnmapMemObject(outSize, pData);

	printf("\nKernel Time: %lf us", ((double)time)/1000.0);
	return time;
}
