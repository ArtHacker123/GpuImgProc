#include "OclCompactPrv.h"

#define OCL_PROGRAM_SOURCE(s) #s

using namespace Ocl;

const char CompactPrv::sSource[] = OCL_PROGRAM_SOURCE(

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

inline void warp_scan(int wid, int i, local volatile int* sh_data)
{
    if (wid >= 1) sh_data[i] += sh_data[i - 1];
    if (wid >= 2) sh_data[i] += sh_data[i - 2];
    if (wid >= 4) sh_data[i] += sh_data[i - 4];
    if (wid >= 8) sh_data[i] += sh_data[i - 8];
    if (WARP_SIZE == 32)
    {
        if (wid >= 16) sh_data[i] += sh_data[i - 16];
    }
}

inline void block_scan(const int i, local volatile int* sh_data)
{
    const int j = i%WARP_SIZE;
    const int k = i / WARP_SIZE;
    const int index = i + (SH_MEM_SIZE / 2);

    warp_scan(j, i, sh_data);
    warp_scan(j, index, sh_data);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i == 0)
    {
        sh_data[SH_MEM_SIZE] = 0;
    }

    if (i < ((SH_MEM_SIZE / WARP_SIZE) - 1))
    {
        sh_data[SH_MEM_SIZE + i + 1] = sh_data[(i*WARP_SIZE) + (WARP_SIZE - 1)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (k == 0)
    {
        warp_scan(j, (SH_MEM_SIZE + i), sh_data);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    sh_data[i] += sh_data[SH_MEM_SIZE + (i / WARP_SIZE)];
    sh_data[index] += sh_data[SH_MEM_SIZE + (index / WARP_SIZE)];
}

kernel void compact_coords_float_x(read_only image2d_t image, float value, global int* p_blk_sum, global int2* p_out, int maxOutSize, global int* p_out_size)
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
            p_out[sh_data[index] - 1] = (int2)(x, y);
        }
    }

    if (read_imagef(image, (int2)(x + 1, y)).x >= value)
    {
        sh_data[index + 1] += sum_blk;
        if (sh_data[index + 1] < maxOutSize)
        {
            p_out[sh_data[index + 1] - 1] = (int2)(x + 1, y);
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
    const int x = 2 * get_global_id(0);
    const int y = get_global_id(1);
    const int index = (get_local_id(1)*get_local_size(0) * 2) + (get_local_id(0) * 2);
    float uv1 = read_imagef(image, (int2)(x, y)).z;
    float uv2 = read_imagef(image, (int2)(x + 1, y)).z;
    sh_data[index] = (uv1 >= value) ? 1 : 0;
    sh_data[index + 1] = (uv2 >= value) ? 1 : 0;
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
            p_out[sh_data[index] - 1] = flowData;
        }
    }

    if (uv2 >= value)
    {
        sh_data[index + 1] += sum_blk;
        if (sh_data[index + 1] < maxOutSize)
        {
            float2 uv = read_imagef(image, (int2)(x + 1, y)).xy;
            OptFlowData flowData;
            flowData.x = x + 1;
            flowData.y = y;
            flowData.u = uv.x;
            flowData.v = uv.y;
            p_out[sh_data[index + 1] - 1] = flowData;
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
