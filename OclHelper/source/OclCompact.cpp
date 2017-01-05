#include "OclCompact.h"
#include "OclUtils.h"

#include <sstream>

#define OCL_PROGRAM_SOURCE(s) #s

using namespace Ocl;

const char Compact::sSource[] = OCL_PROGRAM_SOURCE(

typedef struct _OptFlowData
{
    int x;
    int y;
    float u;
    float v;
} OptFlowData;

typedef struct _HoughData
{
    int rho;
    int angle;
    int strength;
} HoughData;

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

kernel void reduce_sum_float_x(read_only image2d_t image, float value, global int* p_block_sum)
{
    local int sh_data[SH_MEM_SIZE_REDUCE];
    const int x = 4*get_global_id(0);
    const int y = get_global_id(1);
    const int index = (get_local_size(0)*get_local_id(1))+get_local_id(0);
    int data = (read_imagef(image, (int2)(x, y)).x >= value)?1:0;
    data += (read_imagef(image, (int2)(x+1, y)).x >= value)?1:0;
    data += (read_imagef(image, (int2)(x+2, y)).x >= value)?1:0;
    data += (read_imagef(image, (int2)(x+3, y)).x >= value)?1:0;
    sh_data[index] = data;
    barrier(CLK_LOCAL_MEM_FENCE);
    block_reduce_sum(index, sh_data);
    if (index == 0)
    {
        const int xwidth = get_image_width(image)/32;
        const int id = (get_group_id(1)*xwidth)+get_group_id(0);
        p_block_sum[id] = sh_data[0];
    }
}

kernel void reduce_sum_float_z(read_only image2d_t image, float value, global int* p_block_sum)
{
    local int sh_data[SH_MEM_SIZE_REDUCE];
    const int x = 4 * get_global_id(0);
    const int y = get_global_id(1);
    const int index = (get_local_size(0)*get_local_id(1)) + get_local_id(0);
    int data = (read_imagef(image, (int2)(x, y)).z >= value) ? 1 : 0;
    data += (read_imagef(image, (int2)(x + 1, y)).z >= value) ? 1 : 0;
    data += (read_imagef(image, (int2)(x + 2, y)).z >= value) ? 1 : 0;
    data += (read_imagef(image, (int2)(x + 3, y)).z >= value) ? 1 : 0;
    sh_data[index] = data;
    barrier(CLK_LOCAL_MEM_FENCE);
    block_reduce_sum(index, sh_data);
    if (index == 0)
    {
        const int xwidth = get_image_width(image) / 32;
        const int id = (get_group_id(1)*xwidth) + get_group_id(0);
        p_block_sum[id] = sh_data[0];
    }
}

kernel void reduce_sum_int_x(read_only image2d_t image, int value, global int* p_block_sum)
{
    local int sh_data[SH_MEM_SIZE_REDUCE];
    const int x = 4*get_global_id(0);
    const int y = get_global_id(1);
    const int index = (get_local_size(0)*get_local_id(1))+get_local_id(0);
    int data = (read_imageui(image, (int2)(x, y)).x >= value)?1:0;
    data += (read_imageui(image, (int2)(x+1, y)).x >= value)?1:0;
    data += (read_imageui(image, (int2)(x+2, y)).x >= value)?1:0;
    data += (read_imageui(image, (int2)(x+3, y)).x >= value)?1:0;
    sh_data[index] = data;
    barrier(CLK_LOCAL_MEM_FENCE);
    block_reduce_sum(index, sh_data);
    if (index == 0)
    {
        const int xwidth = get_image_width(image)/32;
        const int id = (get_group_id(1)*xwidth)+get_group_id(0);
        p_block_sum[id] = sh_data[0];
    }
}

inline void warp_scan(int wid, int i, local volatile int* sh_data)
{
    if (wid >= 1) sh_data[i] += sh_data[i-1];
    if (wid >= 2) sh_data[i] += sh_data[i-2];
    if (wid >= 4) sh_data[i] += sh_data[i-4];
    if (wid >= 8) sh_data[i] += sh_data[i-8];
    if (WARP_SIZE == 32)
    {
        if (wid >= 16) sh_data[i] += sh_data[i - 16];
    }
}

inline void block_scan(const int i, local volatile int* sh_data)
{
    const int j = i%WARP_SIZE;
    const int k = i/WARP_SIZE;
    const int index = i+(SH_MEM_SIZE/2);

    warp_scan(j, i, sh_data);
    warp_scan(j, index, sh_data);
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

    sh_data[i] += sh_data[SH_MEM_SIZE+(i/WARP_SIZE)];
    sh_data[index] += sh_data[SH_MEM_SIZE+(index/WARP_SIZE)];
}

kernel void compact_coords_float_x(read_only image2d_t image, float value, global int* p_blk_sum, global int2* p_out, int maxOutSize, global int* p_out_size)
{
    local int sum_blk;
    local int sh_data[SH_MEM_SIZE+WARP_SIZE];
    const int x = 2*get_global_id(0);
    const int y = get_global_id(1);
    const int index = (get_local_id(1)*get_local_size(0)*2)+(get_local_id(0)*2);
    sh_data[index] = (read_imagef(image, (int2)(x, y)).x >= value)?1:0;
    sh_data[index+1] = (read_imagef(image, (int2)(x+1, y)).x >= value)?1:0;
    barrier(CLK_LOCAL_MEM_FENCE);
    block_scan((get_local_id(1)*get_local_size(0))+get_local_id(0), sh_data);

    if (index == 0)
    {
        const int xwidth = get_image_width(image)/32;
        const int id = (get_group_id(1)*xwidth) + get_group_id(0);
        sum_blk = (id == 0) ? 0 : p_blk_sum[id-1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_global_id(0) == (get_global_size(0)-1))
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

    if (read_imagef(image, (int2)(x+1, y)).x >= value)
    {
        sh_data[index+1] += sum_blk;
        if (sh_data[index+1] < maxOutSize)
        {
            p_out[sh_data[index+1]-1] = (int2)(x+1, y);
        }
    }
}

kernel void compact_cartesian_coords_float_x(read_only image2d_t image, float value, global int* p_blk_sum, global int2* p_out, int maxOutSize, global int* p_out_size)
{
    local int sum_blk;
    local int sh_data[SH_MEM_SIZE + WARP_SIZE];
    const int x = 2 * get_global_id(0);
    const int y = get_global_id(1);
    const int index = (get_local_id(1)*get_local_size(0) * 2) + (get_local_id(0) * 2);
    sh_data[index] = (read_imagef(image, (int2)(x, y)).x >= value) ? 1 : 0;
    sh_data[index + 1] = (read_imagef(image, (int2)(x + 1, y)).x >= value) ? 1 : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    block_scan((get_local_id(1)*get_local_size(0)) + get_local_id(0), sh_data);

    if (index == 0)
    {
        const int xwidth = get_image_width(image) / 32;
        const int id = (get_group_id(1)*xwidth) + get_group_id(0);
        sum_blk = (id == 0) ? 0 : p_blk_sum[id - 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_global_id(0) == (get_global_size(0) - 1))
    {
        int count = sh_data[SH_MEM_SIZE - 1] + sum_blk;
        p_out_size[0] = (maxOutSize > count) ? count : maxOutSize;
    }

    if (read_imagef(image, (int2)(x, y)).x >= value)
    {
        sh_data[index] += sum_blk;
        if (sh_data[index] < maxOutSize)
        {
            p_out[sh_data[index] - 1] = (int2)(x - (get_image_width(image) / 2), (get_image_height(image) / 2) - y);
        }
    }

    if (read_imagef(image, (int2)(x + 1, y)).x >= value)
    {
        sh_data[index + 1] += sum_blk;
        if (sh_data[index + 1] < maxOutSize)
        {
            p_out[sh_data[index + 1] - 1] = (int2)(x + 1 - (get_image_width(image) / 2), (get_image_height(image) / 2) - y);
        }
    }
}

kernel void compact_optflow(read_only image2d_t image, float value, global int* p_blk_sum, global OptFlowData* p_out, int maxOutSize, global int* p_out_size)
{
    local int sum_blk;
    local int sh_data[SH_MEM_SIZE + WARP_SIZE];
    const int x = 2*get_global_id(0);
    const int y = get_global_id(1);
    const int index = (get_local_id(1)*get_local_size(0)*2)+(get_local_id(0)*2);
    float uv1 = read_imagef(image, (int2)(x, y)).z;
    float uv2 = read_imagef(image, (int2)(x+1, y)).z;
    sh_data[index] = (uv1 >= value) ? 1 : 0;
    sh_data[index+1] = (uv2 >= value) ? 1 : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    block_scan((get_local_id(1)*get_local_size(0))+get_local_id(0), sh_data);

    if (index == 0)
    {
        const int xwidth = get_image_width(image) / 32;
        const int id = (get_group_id(1)*xwidth) + get_group_id(0);
        sum_blk = (id == 0) ? 0 : p_blk_sum[id-1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_global_id(0) == (get_global_size(0) - 1))
    {
        int count = sh_data[SH_MEM_SIZE - 1] + sum_blk;
        p_out_size[0] = (maxOutSize > count) ? count : maxOutSize;
    }

    if (uv1 >= value)
    {
        sh_data[index] += sum_blk;
        if (sh_data[index] < maxOutSize)
        {
            float2 uv = read_imagef(image, (int2)(x, y)).xy;
            OptFlowData flowData;
            flowData.x = x;
            flowData.y = y;
            flowData.u = uv.x;
            flowData.v = uv.y;
            p_out[sh_data[index]-1] = flowData;
        }
    }

    if (uv2 >= value)
    {
        sh_data[index+1] += sum_blk;
        if (sh_data[index+1] < maxOutSize)
        {
            float2 uv = read_imagef(image, (int2)(x+1, y)).xy;
            OptFlowData flowData;
            flowData.x = x + 1;
            flowData.y = y;
            flowData.u = uv.x;
            flowData.v = uv.y;
            p_out[sh_data[index+1]-1] = flowData;
        }
    }
}

kernel void compact_hough_data(read_only image2d_t image, int value, global int* p_blk_sum, global HoughData* p_out, int maxOutSize, global int* p_out_size)
{
    local int sum_blk;
    local int sh_data[SH_MEM_SIZE + WARP_SIZE];
    const int x = 2 * get_global_id(0);
    const int y = get_global_id(1);
    const int index = (get_local_id(1)*get_local_size(0) * 2) + (get_local_id(0) * 2);
    sh_data[index] = (read_imageui(image, (int2)(x, y)).x >= value) ? 1 : 0;
    sh_data[index + 1] = (read_imageui(image, (int2)(x + 1, y)).x >= value) ? 1 : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    block_scan((get_local_id(1)*get_local_size(0)) + get_local_id(0), sh_data);

    if (index == 0)
    {
        const int xwidth = get_image_width(image) / 32;
        const int id = (get_group_id(1)*xwidth) + get_group_id(0);
        sum_blk = (id == 0) ? 0 : p_blk_sum[id - 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_global_id(0) == (get_global_size(0) - 1))
    {
        int count = sh_data[SH_MEM_SIZE - 1] + sum_blk;
        p_out_size[0] = (maxOutSize > count) ? count : maxOutSize;
    }

    if (read_imageui(image, (int2)(x, y)).x >= value)
    {
        sh_data[index] += sum_blk;
        if (sh_data[index] < maxOutSize)
        {
            HoughData hdata;
            hdata.rho = x;
            hdata.angle = y;
            hdata.strength = read_imageui(image, (int2)(x, y)).x;
            p_out[sh_data[index] - 1] = hdata;
        }
    }

    if (read_imageui(image, (int2)(x + 1, y)).x >= value)
    {
        sh_data[index + 1] += sum_blk;
        if (sh_data[index + 1] < maxOutSize)
        {
            HoughData hdata;
            hdata.rho = x + 1;
            hdata.angle = y;
            hdata.strength = read_imageui(image, (int2)(x + 1, y)).x;
            p_out[sh_data[index + 1] - 1] = hdata;
        }
    }
}

);

Compact::Compact(cl::Context& ctxt, cl::CommandQueue& queue)
    :mContext(ctxt),
     mQueue(queue),
     mScanBlkSize(256),
     mReduceBlkSize(64),
     mOutSize(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 1),
     mScan(ctxt, queue)
{
    try
    {
        init(32);
        size_t wgmSize = Ocl::getWorkGroupSizeMultiple(mQueue, mCompactFloatX);
        //Intel HD4xxx series GPU's warp_size is not 32
        if (wgmSize != 32)
        {
            //re-initialize with wgmSize
            init(wgmSize);
        }
    }

    catch (cl::Error error)
    {
        fprintf(stderr, "Error: %s", error.what());
        exit(0);
    }
}

Compact::~Compact()
{
}

void Compact::init(int warp_size)
{
    std::ostringstream options;
    options << "-DSH_MEM_SIZE_REDUCE=" << 64 << " -DSH_MEM_SIZE=" << 256 << " -DWARP_SIZE=" << warp_size;

    cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
    mProgram = cl::Program(mContext, source);
    mProgram.build(options.str().c_str());

    mReduceFloatX = cl::Kernel(mProgram, "reduce_sum_float_x");
    mCompactFloatX = cl::Kernel(mProgram, "compact_coords_float_x");
    mCompactCartFloatX = cl::Kernel(mProgram, "compact_cartesian_coords_float_x");

    mReduceFloatZ = cl::Kernel(mProgram, "reduce_sum_float_z");
    mCompactOptFlow = cl::Kernel(mProgram, "compact_optflow");

    mReduceIntX = cl::Kernel(mProgram, "reduce_sum_int_x");
    mCompactHoughData = cl::Kernel(mProgram, "compact_hough_data");
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

size_t Compact::process(const cl::Image& inpImage, Ocl::DataBuffer<Ocl::Pos>& out, float value, size_t& outCount)
{
    size_t width, height;
    size_t maxOutSize = out.count();
    inpImage.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
    inpImage.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);

    createIntBuffer(width*height);

    mReduceFloatX.setArg(0, inpImage);
    mReduceFloatX.setArg(1, value);
    mReduceFloatX.setArg(2, *mBuffReduce);
    cl::Event event;
    mQueue.enqueueNDRangeKernel(mReduceFloatX, cl::NullRange, cl::NDRange(width/4, height), cl::NDRange(8, 8), NULL, &event);
    event.wait();
    size_t time = kernelExecTime(mQueue, event);

    time += mScan.process(*mBuffReduce);

    mCompactFloatX.setArg(0, inpImage);
    mCompactFloatX.setArg(1, value);
    mCompactFloatX.setArg(2, *mBuffReduce);
    mCompactFloatX.setArg(3, out.buffer());
    mCompactFloatX.setArg(4, (int)maxOutSize);
    mCompactFloatX.setArg(5, mOutSize);
    mQueue.enqueueNDRangeKernel(mCompactFloatX, cl::NullRange, cl::NDRange(width/2, height), cl::NDRange(16, 8), NULL, &event);
    event.wait();
    time += kernelExecTime(mQueue, event);

    int* pData = mOutSize.map(mQueue, CL_TRUE, CL_MAP_READ, 0, 1);
    outCount = *pData;
    mOutSize.unmap(mQueue, pData);
    return time;
}

size_t Compact::process_cartesian(const cl::Image& inpImage, Ocl::DataBuffer<Ocl::Pos>& coords, float threshold, size_t& count)
{
    size_t width, height;
    size_t maxOutSize = coords.count();
    inpImage.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
    inpImage.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);

    createIntBuffer(width*height);

    mReduceFloatX.setArg(0, inpImage);
    mReduceFloatX.setArg(1, threshold);
    mReduceFloatX.setArg(2, *mBuffReduce);
    cl::Event event;
    mQueue.enqueueNDRangeKernel(mReduceFloatX, cl::NullRange, cl::NDRange(width/4, height), cl::NDRange(8, 8), NULL, &event);
    event.wait();
    size_t time = kernelExecTime(mQueue, event);

    time += mScan.process(*mBuffReduce);

    mCompactCartFloatX.setArg(0, inpImage);
    mCompactCartFloatX.setArg(1, threshold);
    mCompactCartFloatX.setArg(2, *mBuffReduce);
    mCompactCartFloatX.setArg(3, coords.buffer());
    mCompactCartFloatX.setArg(4, (int)maxOutSize);
    mCompactCartFloatX.setArg(5, mOutSize);
    mQueue.enqueueNDRangeKernel(mCompactCartFloatX, cl::NullRange, cl::NDRange(width/2, height), cl::NDRange(16, 8), NULL, &event);
    event.wait();
    time += kernelExecTime(mQueue, event);

    int* pData = mOutSize.map(mQueue, CL_TRUE, CL_MAP_READ, 0, 1);
    count = *pData;
    mOutSize.unmap(mQueue, pData);
    return time;
}

size_t Compact::process(const cl::Image& inpImage, Ocl::DataBuffer<Ocl::OptFlowData>& flowData, float threshold, size_t& count)
{
    size_t width, height;
    size_t maxOutSize = flowData.count();
    inpImage.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
    inpImage.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);

    createIntBuffer(width*height);

    mReduceFloatZ.setArg(0, inpImage);
    mReduceFloatZ.setArg(1, threshold);
    mReduceFloatZ.setArg(2, *mBuffReduce);
    cl::Event event;
    mQueue.enqueueNDRangeKernel(mReduceFloatZ, cl::NullRange, cl::NDRange(width/4, height), cl::NDRange(8, 8), NULL, &event);
    event.wait();
    size_t time = kernelExecTime(mQueue, event);

    time += mScan.process(*mBuffReduce);

    mCompactOptFlow.setArg(0, inpImage);
    mCompactOptFlow.setArg(1, threshold);
    mCompactOptFlow.setArg(2, *mBuffReduce);
    mCompactOptFlow.setArg(3, flowData.buffer());
    mCompactOptFlow.setArg(4, (int)maxOutSize);
    mCompactOptFlow.setArg(5, mOutSize);

    mQueue.enqueueNDRangeKernel(mCompactOptFlow, cl::NullRange, cl::NDRange(width/2, height), cl::NDRange(16, 8), NULL, &event);
    event.wait();
    time += kernelExecTime(mQueue, event);

    int* pData = mOutSize.map(mQueue, CL_TRUE, CL_MAP_READ, 0, 1);
    count = *pData;
    mOutSize.unmap(mQueue, pData);
    return time;
}

size_t Compact::process(const cl::Image& inpImage, Ocl::DataBuffer<Ocl::HoughData>& houghData, size_t threshold, size_t& count)
{
    size_t width, height;
    size_t maxOutSize = houghData.count();
    inpImage.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
    inpImage.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);

    createIntBuffer(width*height);

    mReduceIntX.setArg(0, inpImage);
    mReduceIntX.setArg(1, (int)threshold);
    mReduceIntX.setArg(2, *mBuffReduce);
    cl::Event event;
    mQueue.enqueueNDRangeKernel(mReduceIntX, cl::NullRange, cl::NDRange(width/4, height), cl::NDRange(8, 8), NULL, &event);
    event.wait();
    size_t time = kernelExecTime(mQueue, event);

    time += mScan.process(*mBuffReduce);

    mCompactHoughData.setArg(0, inpImage);
    mCompactHoughData.setArg(1, (int)threshold);
    mCompactHoughData.setArg(2, *mBuffReduce);
    mCompactHoughData.setArg(3, houghData.buffer());
    mCompactHoughData.setArg(4, (int)maxOutSize);
    mCompactHoughData.setArg(5, mOutSize);

    mQueue.enqueueNDRangeKernel(mCompactHoughData, cl::NullRange, cl::NDRange(width/2, height), cl::NDRange(16, 8), NULL, &event);
    event.wait();
    time += kernelExecTime(mQueue, event);

    int* pData = mOutSize.map(mQueue, CL_TRUE, CL_MAP_READ, 0, 1);
    count = *pData;
    mOutSize.unmap(mQueue, pData);
    return time;
}