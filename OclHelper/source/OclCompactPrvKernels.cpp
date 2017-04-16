#include "OclCompactPrv.h"

#define OCL_PROGRAM_SOURCE(s) #s

using namespace Ocl;

const std::string CompactPrv::sSource = OCL_PROGRAM_SOURCE(

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
    if (index < 32) sh_data[index] += sh_data[32 + index];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (index < 16) sh_data[index] += sh_data[16 + index];
    if (index < 8) sh_data[index] += sh_data[8 + index];
    if (index < 4) sh_data[index] += sh_data[4 + index];
    if (index < 2) sh_data[index] += sh_data[2 + index];
    if (index < 1) sh_data[index] += sh_data[1 + index];
    barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void reduce_sum_float_x(read_only image2d_t image, float value, global int* p_block_sum)
{
    local int sh_data[SH_MEM_SIZE_REDUCE];
    const int x = 4 * get_global_id(0);
    const int y = get_global_id(1);
    const int index = (get_local_size(0)*get_local_id(1)) + get_local_id(0);
    int data = (read_imagef(image, (int2)(x, y)).x >= value) ? 1 : 0;
    data += (read_imagef(image, (int2)(x + 1, y)).x >= value) ? 1 : 0;
    data += (read_imagef(image, (int2)(x + 2, y)).x >= value) ? 1 : 0;
    data += (read_imagef(image, (int2)(x + 3, y)).x >= value) ? 1 : 0;
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
    const int x = 4 * get_global_id(0);
    const int y = get_global_id(1);
    const int index = (get_local_size(0)*get_local_id(1)) + get_local_id(0);
    int data = (read_imageui(image, (int2)(x, y)).x >= value) ? 1 : 0;
    data += (read_imageui(image, (int2)(x + 1, y)).x >= value) ? 1 : 0;
    data += (read_imageui(image, (int2)(x + 2, y)).x >= value) ? 1 : 0;
    data += (read_imageui(image, (int2)(x + 3, y)).x >= value) ? 1 : 0;
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

inline void wscan(int i, local volatile int* sh_data)
{
    const int wid = i&(WARP_SIZE-1);
    if (wid >= 1) sh_data[i] += sh_data[i-1];
    if (wid >= 2) sh_data[i] += sh_data[i-2];
    if (wid >= 4) sh_data[i] += sh_data[i-4];
    if (wid >= 8) sh_data[i] += sh_data[i-8];
    if (WARP_SIZE == 32)
    {
        if (wid >= 16) sh_data[i] += sh_data[i-16];
    }
}

inline void bscan(int i, local volatile int* shData, local volatile int* shDataT)
{
    wscan(i, shData);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < (SH_MEM_SIZE/WARP_SIZE)) shDataT[i] = shData[(WARP_SIZE*i)+(WARP_SIZE-1)];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < WARP_SIZE) wscan(i, shDataT);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i >= WARP_SIZE) shData[i] += shDataT[(i/WARP_SIZE)-1];
    barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void compact_coords_float_x(read_only image2d_t image, float value, global int* p_blk_sum, global int2* p_out, int maxOutSize, global int* p_out_size)
{
    local int sum_blk;
    local int sh_data[SH_MEM_SIZE+WARP_SIZE];

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    int i = (get_local_id(1)*get_local_size(0))+get_local_id(0);
    sh_data[i] = (read_imagef(image, (int2)(x, y)).x >= value)?1:0;
    bscan(i, sh_data, &sh_data[SH_MEM_SIZE]);

    if (i == 0)
    {
        const int id = (get_group_id(1)*get_num_groups(0))+get_group_id(0);
        sum_blk = (id == 0)?0:p_blk_sum[id-1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (get_global_id(0) == (get_global_size(0) - 1))
    {
        int count = sh_data[SH_MEM_SIZE-1] + sum_blk;
        p_out_size[0] = (maxOutSize > count)?count:maxOutSize;
    }

    if (read_imagef(image, (int2)(x, y)).x >= value)
    {
        sh_data[i] += sum_blk;
        if (sh_data[i] < maxOutSize) p_out[sh_data[i]-1] = (int2)(x, y);
    }
}

kernel void compact_cartesian_coords_float_x(read_only image2d_t image, float value, global int* p_blk_sum, global int2* p_out, int maxOutSize, global int* p_out_size)
{
    local int sum_blk;
    local int sh_data[SH_MEM_SIZE+WARP_SIZE];

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    int i = (get_local_id(1)*get_local_size(0))+get_local_id(0);
    sh_data[i] = (read_imagef(image, (int2)(x, y)).x >= value) ? 1 : 0;
    bscan(i, sh_data, sh_data[SH_MEM_SIZE]);

    if (i == 0)
    {
        const int id = (get_group_id(1)*get_num_groups(0))+get_group_id(0);
        sum_blk = (id == 0) ? 0 : p_blk_sum[id-1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_global_id(0) == (get_global_size(0) - 1))
    {
        int count = sh_data[SH_MEM_SIZE-1] + sum_blk;
        p_out_size[0] = (maxOutSize > count) ? count : maxOutSize;
    }

    if (read_imagef(image, (int2)(x, y)).x >= value)
    {
        sh_data[i] += sum_blk;
        if (sh_data[i] < maxOutSize)
        {
            p_out[sh_data[i]-1] = (int2)(x-(get_image_width(image)/2), (get_image_height(image)/2)-y);
        }
    }
}

kernel void compact_optflow(read_only image2d_t image, float value, global int* p_blk_sum, global OptFlowData* p_out, int maxOutSize, global int* p_out_size)
{
    local int sum_blk;
    local int sh_data[SH_MEM_SIZE + WARP_SIZE];
    
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    int i = (get_local_id(1)*get_local_size(0))+get_local_id(0);
    sh_data[i] = (read_imagef(image, (int2)(x, y)).z >= value) ? 1 : 0;
    bscan(i, sh_data, &sh_data[SH_MEM_SIZE]);

    if (i == 0)
    {
        const int id = (get_group_id(1)*get_num_groups(0))+get_group_id(0);
        sum_blk = (id == 0) ? 0 : p_blk_sum[id-1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_global_id(0) == (get_global_size(0)-1))
    {
        int count = sh_data[SH_MEM_SIZE-1] + sum_blk;
        p_out_size[0] = (maxOutSize > count) ? count : maxOutSize;
    }

    if (read_imagef(image, (int2)(x, y)).z >= value)
    {
        sh_data[i] += sum_blk;
        if (sh_data[i] < maxOutSize)
        {
            float2 uv = read_imagef(image, (int2)(x, y)).xy;
            OptFlowData flowData;
            flowData.x = x;
            flowData.y = y;
            flowData.u = uv.x;
            flowData.v = uv.y;
            p_out[sh_data[i]-1] = flowData;
        }
    }
}

kernel void compact_hough_data(read_only image2d_t image, int value, global int* p_blk_sum, global HoughData* p_out, int maxOutSize, global int* p_out_size)
{
    local int sum_blk;
    local int sh_data[SH_MEM_SIZE+WARP_SIZE];

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    int i = (get_local_id(1)*get_local_size(0))+get_local_id(0);
    sh_data[i] = (read_imageui(image, (int2)(x, y)).x >= value) ? 1 : 0;
    bscan(i, sh_data, &sh_data[SH_MEM_SIZE]);

    if (i == 0)
    {
        const int id = (get_group_id(1)*get_num_groups(0))+get_group_id(0);
        sum_blk = (id == 0) ? 0 : p_blk_sum[id - 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_global_id(0) == (get_global_size(0)-1))
    {
        int count = sh_data[SH_MEM_SIZE-1] + sum_blk;
        p_out_size[0] = (maxOutSize > count) ? count : maxOutSize;
    }

    if (read_imageui(image, (int2)(x, y)).x >= value)
    {
        sh_data[i] += sum_blk;
        if (sh_data[i] < maxOutSize)
        {
            HoughData hdata;
            hdata.rho = x;
            hdata.angle = y;
            hdata.strength = read_imageui(image, (int2)(x, y)).x;
            p_out[sh_data[i]-1] = hdata;
        }
    }
}

kernel void scan(global const int* pInput, global int* pOutput, const int count)
{
    local int shData[SH_MEM_SIZE+WARP_SIZE];
    int i = get_local_id(0);
    int id = get_global_id(0);
    shData[i] = (id < count)?pInput[id]:0;
    bscan(i, shData, &shData[SH_MEM_SIZE]);
    if (id < count) pOutput[id] = shData[i];
}

kernel void gather(global const int* pInput, global int* pOutput, const int start, const int count)
{
    local int shData[SH_MEM_SIZE+WARP_SIZE];
    int i = get_local_id(0);
    int id = start+(get_local_size(0)*get_local_id(0));
    shData[i] = (id < count)?pInput[id]:0;
    bscan(i, shData, &shData[SH_MEM_SIZE]);
    pOutput[i] = shData[i];
}

kernel void add(global const int* pInput, global int* pOutput, const int start, const int count)
{
    local int shData;
    if (get_local_id(0) == 0) shData = pInput[get_group_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    const int id = start+get_global_id(0);
    if (id < count) pOutput[id] += shData;
}

);
