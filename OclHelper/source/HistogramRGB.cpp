#include "HistogramRGB.h"

#define LSIZE_X 16
#define LSIZE_Y 16

#define OCL_PROGRAM_SOURCE(s) #s

using namespace Ocl;

const char HistogramRGB::sSource[] = OCL_PROGRAM_SOURCE(
kernel void histogram_temp_rgb_float(read_only image2d_t image, global unsigned int *hist_data)
{
	local unsigned int sh_data[3*256];
	unsigned int lindex = (get_local_id(1)*get_local_size(0)) + get_local_id(0);
	if (lindex < 256)
	{
		sh_data[lindex] = 0;
		sh_data[256+lindex] = 0;
		sh_data[512+lindex] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int2 coord;
	coord.x = (get_global_id(0)<<2);
	coord.y = get_global_id(1)<<2;
	if (coord.x < get_image_width(image) && coord.y < get_image_height(image))
	{
		for (int i = 0; i < 4; i++)
		{
			coord.y = get_global_id(1)<<2;
			for (int j = 0; j < 4; j++)
			{
				atomic_inc(&sh_data[(int)ceil(255.0*read_imagef(image, coord).x)]);
				atomic_inc(&sh_data[256+(int)ceil(255.0*read_imagef(image, coord).y)]);
				atomic_inc(&sh_data[512+(int)ceil(255.0*read_imagef(image, coord).z)]);
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
        int off_index = (max_offset*lindex) + offset;
        hist_data[off_index] = sh_data[lindex];
        off_index += (max_offset * 256);
        hist_data[off_index] = sh_data[256 + lindex];
        off_index += (max_offset * 256);
        hist_data[off_index] = sh_data[512 + lindex];
	}
	//barrier(CLK_GLOBAL_MEM_FENCE);
}

kernel void histogram_temp_rgb_uint8(read_only image2d_t image, global unsigned int *hist_data)
{
    local unsigned int sh_data[3*256];
    unsigned int lindex = (get_local_id(1)*get_local_size(0)) + get_local_id(0);
    if (lindex < 256)
    {
        sh_data[lindex] = 0;
        sh_data[256 + lindex] = 0;
        sh_data[512 + lindex] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int2 coord;
    coord.x = (get_global_id(0) << 2);
    coord.y = get_global_id(1) << 2;
    if (coord.x < get_image_width(image) && coord.y < get_image_height(image))
    {
        for (int i = 0; i < 4; i++)
        {
            coord.y = get_global_id(1) << 2;
            for (int j = 0; j < 4; j++)
            {
                atomic_inc(&sh_data[read_imageui(image, coord).x]);
                atomic_inc(&sh_data[256 + read_imageui(image, coord).y]);
                atomic_inc(&sh_data[512 + read_imageui(image, coord).z]);
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
        int off_index = (max_offset*lindex)+offset;
        hist_data[off_index] = sh_data[lindex];
        off_index += (max_offset*256);
        hist_data[off_index] = sh_data[256+lindex];
        off_index += (max_offset*256);
        hist_data[off_index] = sh_data[512+lindex];
    }
    //barrier(CLK_GLOBAL_MEM_FENCE);
}

inline void block_reduce_sum(local int* sh_data)
{
    if (get_local_id(0) < 32) sh_data[get_local_id(0)] += sh_data[32 + get_local_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) < 16) sh_data[get_local_id(0)] += sh_data[16 + get_local_id(0)];
    if (get_local_id(0) < 8) sh_data[get_local_id(0)] += sh_data[8 + get_local_id(0)];
    if (get_local_id(0) < 4) sh_data[get_local_id(0)] += sh_data[4 + get_local_id(0)];
    if (get_local_id(0) < 2) sh_data[get_local_id(0)] += sh_data[2 + get_local_id(0)];
    if (get_local_id(0) < 1) sh_data[get_local_id(0)] += sh_data[1 + get_local_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void accum_histogram_rgb(global read_only unsigned int *p_int_hist_data, global unsigned int *p_hist_data, size_t count)
{
    local int sh_data[64];
    int lid = get_local_id(0);
    int index = (get_group_id(0)*count);
    sh_data[get_local_id(0)] = 0;
    while (lid < count)
    {
        sh_data[get_local_id(0)] += p_int_hist_data[index+lid];
        lid += get_local_size(0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    block_reduce_sum(sh_data);
    if (get_local_id(0) == 0)
    {
        p_hist_data[get_group_id(0)] = sh_data[0];
    }
}

);

HistogramRGB::HistogramRGB(cl::Context& ctxt, cl::CommandQueue& queue)
	:mContext(ctxt),
	 mQueue(queue)
{
    try
    {
        cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
        mPgm = cl::Program(mContext, source);
        mPgm.build();

        // create the kernel
        mAccHist = cl::Kernel(mPgm, "accum_histogram_rgb");
        mTempHistFloat = cl::Kernel(mPgm, "histogram_temp_rgb_float");
        mTempHistUint8 = cl::Kernel(mPgm, "histogram_temp_rgb_uint8");
    }

    catch (cl::Error error)
    {
        fprintf(stderr, "%s", error.what());
        exit(0);
    }

    /*cl_device_id devId = 0;
    queue.getInfo<cl_device_id>(CL_QUEUE_DEVICE, &devId);

    size_t gSize = 0;
    cl::Device device(devId);
    std::string name;
    device.getInfo<std::string>(CL_DEVICE_NAME, &name);
    mAccHist.getWorkGroupInfo<size_t>(device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &gSize);*/
}

HistogramRGB::~HistogramRGB()
{
}

void HistogramRGB::createTempHistBuffer(size_t size)
{
    size_t memSize = (mTempBuff.get() == 0)?0:mTempBuff->count();
    if (memSize != size)
    {
        mTempBuff.reset(new Ocl::DataBuffer<int>(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, size));
    }
}

size_t HistogramRGB::computeTempHist(const cl::Image& image, size_t& count)
{
    size_t width, height;

    image.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
    image.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);

    size_t wsize0, wsize1;
    wsize0 = ((width%(LSIZE_X<<2)) == 0) ? (width/4) : ((width + (LSIZE_X<<2) - ((width % (LSIZE_X << 2)))) / 4);
    wsize1 = ((height%(LSIZE_Y<<2)) == 0) ? (height/4) : ((height + (LSIZE_Y<<2) - ((height % (LSIZE_Y << 2)))) / 4);

    count = (wsize0/LSIZE_X)*(wsize1/LSIZE_Y);
    size_t sizeTempHist = 3*256*count;
    createTempHistBuffer(sizeTempHist);

    size_t time = 0;
    cl_image_format format;
    image.getImageInfo<cl_image_format>(CL_IMAGE_FORMAT, &format);

    if (format.image_channel_order == CL_RGBA)
    {
        cl::Event event;
        switch (format.image_channel_data_type)
        {
            case CL_FLOAT:
            case CL_UNORM_INT8:
                // set the kernel arguments
                mTempHistFloat.setArg(0, image);
                mTempHistFloat.setArg(1, *mTempBuff);
                mQueue.enqueueNDRangeKernel(mTempHistFloat, cl::NullRange, cl::NDRange(wsize0, wsize1), cl::NDRange(LSIZE_X, LSIZE_Y), NULL, &event);
                event.wait();
                time = (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
                break;

            case CL_UNSIGNED_INT8:
                // set the kernel arguments
                mTempHistUint8.setArg(0, image);
                mTempHistUint8.setArg(1, *mTempBuff);
                mQueue.enqueueNDRangeKernel(mTempHistUint8, cl::NullRange, cl::NDRange(wsize0, wsize1), cl::NDRange(LSIZE_X, LSIZE_Y), NULL, &event);
                event.wait();
                time = (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
                break;
        }
    }
    return time;
}

size_t HistogramRGB::accumTempHist(size_t count, Ocl::DataBuffer<int>& rgbBins)
{
    mAccHist.setArg(0, mTempBuff->buffer());
    mAccHist.setArg(1, rgbBins.buffer());
    mAccHist.setArg(2, count);

    cl::Event event;
    mQueue.enqueueNDRangeKernel(mAccHist, cl::NullRange, cl::NDRange(3*256*64), cl::NDRange(64), NULL, &event);
    event.wait();

    size_t time = (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

    return time;
}

size_t HistogramRGB::compute(const cl::ImageGL& image, Ocl::DataBuffer<int>& rgbBins)
{
    size_t count = 0;
	std::vector<cl::Memory> gl_objs = { image };
	mQueue.enqueueAcquireGLObjects(&gl_objs);
    size_t time = computeTempHist(image, count);
	mQueue.enqueueReleaseGLObjects(&gl_objs);
    time += accumTempHist(count, rgbBins);
	return time;
}

size_t HistogramRGB::compute(const cl::Image2D& image, Ocl::DataBuffer<int>& rgbBins)
{
    size_t count = 0;
    size_t time = computeTempHist(image, count);
    time += accumTempHist(count, rgbBins);
    return time;
}
