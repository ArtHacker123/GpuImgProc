#include "OclHistogram.h"

#define LSIZE_X 32
#define LSIZE_Y 16

#define OCL_PROGRAM_SOURCE(s) #s

const char OclHistogram::sHistPgmSrc[] = OCL_PROGRAM_SOURCE(
kernel void IntHistogram(read_only image2d_t image, global unsigned int *hist_data)
{
	__local unsigned int sh_data[256];
	unsigned int lindex = (get_local_id(1)*get_local_size(0)) + get_local_id(0);
	if (lindex < 256)
	{
		sh_data[lindex] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);


	int2 coord;
	coord.x = (get_global_id(0)<<2);
	coord.y = (get_global_id(1)<<2);
	if (coord.x < get_image_width(image) && coord.y < get_image_height(image))
	{
		for (int i = 0; i < 4; i++)
		{
			coord.y = get_global_id(1)<<2;
			for (int j = 0; j < 4; j++)
			{
				atomic_inc(&sh_data[read_imageui(image, coord).x]);
				++coord.y;
			}
			++coord.x;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	unsigned int max_offset = get_num_groups(0)*get_num_groups(1);
	unsigned int offset = (get_group_id(1)*get_num_groups(0)) + get_group_id(0);
	if (lindex < 256 && offset < max_offset)
	{
		hist_data[(256 * offset) + lindex] = sh_data[lindex];
	}
	//barrier(CLK_GLOBAL_MEM_FENCE);
}

kernel void AccHistogram(global read_only unsigned int *p_int_hist_data, global unsigned int *p_hist_data, size_t count)
{
	unsigned int data = 0;
	for (size_t i = 0; i < count; i++)
	{
		data += p_int_hist_data[(256*i) + get_global_id(0)];
	}
	p_hist_data[get_global_id(0)] = data;
});

OclHistogram::OclHistogram(cl::Context& ctxt, cl::CommandQueue& q)
	:mContext(ctxt),
	 mQueue(q)
{
	cl::Program::Sources source(1, std::make_pair(sHistPgmSrc, strlen(sHistPgmSrc)));
	mPgm = cl::Program(mContext, source);
	mPgm.build();

	// create the kernel
	mIntHistKernel = cl::Kernel(mPgm, "IntHistogram");
	mAccHistKernel = cl::Kernel(mPgm, "AccHistogram");
}

OclHistogram::~OclHistogram()
{
}

void OclHistogram::compute(const cl::Image2D& img, cl::Buffer& histBins)
{
	size_t width, height;
	size_t wsize0, wsize1;

	img.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
	img.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);

	if ((width%(LSIZE_X<<2)) == 0)
	{
		wsize0 = width/4;
	}
	else
	{
		wsize0 = (width+(LSIZE_X<<2)-((width%(LSIZE_X<<2))))/4;
	}

	if ((height%(LSIZE_Y<<2)) == 0)
	{
		wsize1 = height/4;
	}
	else
	{
		wsize1 = (height+(LSIZE_Y<<2)-((height%(LSIZE_Y<<2))))/4;
	}

	size_t memSize = 0;
	size_t count = (wsize0/LSIZE_X)*(wsize1/LSIZE_Y);
	size_t sizeIntHist = 256*count*sizeof(int);
	if (mTempBuff.get() != 0)
	{
		mTempBuff->getInfo<size_t>(CL_MEM_SIZE, &memSize);
	}

	if (memSize != sizeIntHist)
	{
		mTempBuff.reset(new cl::Buffer(mContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeIntHist));
	}

	// set the kernel arguments
	mIntHistKernel.setArg(0, img);
	mIntHistKernel.setArg(1, *mTempBuff);
	cl::Event event1;
	mQueue.enqueueNDRangeKernel(mIntHistKernel, cl::NullRange, cl::NDRange(wsize0, wsize1), cl::NDRange(LSIZE_X, LSIZE_Y), NULL, &event1);
	event1.wait();

	mAccHistKernel.setArg(0, *mTempBuff);
	mAccHistKernel.setArg(1, histBins);
	mAccHistKernel.setArg(2, count);
	cl::Event event2;
	mQueue.enqueueNDRangeKernel(mAccHistKernel, cl::NullRange, cl::NDRange(256), cl::NDRange(1), NULL, &event2);
	event2.wait();

	cl_int time1 = (event1.getProfilingInfo<CL_PROFILING_COMMAND_END>()-event1.getProfilingInfo<CL_PROFILING_COMMAND_START>());
	cl_int time2 = (event2.getProfilingInfo<CL_PROFILING_COMMAND_END>()-event2.getProfilingInfo<CL_PROFILING_COMMAND_START>());

	double tot_time = ((double)(time1+time2))/1000000.0;
	printf("\nKernel Time: %lf ms", tot_time);
}
