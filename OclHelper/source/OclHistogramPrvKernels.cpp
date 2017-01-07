#include "OclHistogramPrv.h"

#define OCL_PROGRAM_SOURCE(s) #s

using namespace Ocl;

const char HistogramPrv::sSource[] = OCL_PROGRAM_SOURCE(

kernel void histogram_temp_rgb_float(read_only image2d_t image, global unsigned int *hist_data)
{
    local int sh_data[3 * 256];
    int lindex = (get_local_id(1)*get_local_size(0)) + get_local_id(0);
    if (lindex < 256)
    {
        sh_data[lindex] = 0;
        sh_data[256 + lindex] = 0;
        sh_data[512 + lindex] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int i = get_global_id(0);
    while (i < get_image_width(image))
    {
        float3 color = ceil(255.0f*read_imagef(image, (int2)(i, get_global_id(1))).xyz);
        atomic_inc(&sh_data[(int)color.x]);
        atomic_inc(&sh_data[256 + (int)color.y]);
        atomic_inc(&sh_data[512 + (int)color.z]);
        i += get_global_size(0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int max_offset = get_num_groups(0)*get_num_groups(1);
    int offset = (get_group_id(1)*get_num_groups(0)) + get_group_id(0);
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
    local int sh_data[3 * 256];
    int lindex = (get_local_id(1)*get_local_size(0)) + get_local_id(0);
    if (lindex < 256)
    {
        sh_data[lindex] = 0;
        sh_data[256 + lindex] = 0;
        sh_data[512 + lindex] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int i = get_global_id(0);
    while (i < get_image_width(image))
    {
        int2 coord = (int2)(i, get_global_id(1));
        uint3 color = read_imageui(image, coord).xyz;
        atomic_inc(&sh_data[color.x]);
        atomic_inc(&sh_data[256 + color.y]);
        atomic_inc(&sh_data[512 + color.z]);
        i += get_global_size(0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int max_offset = get_num_groups(0)*get_num_groups(1);
    int offset = (get_group_id(1)*get_num_groups(0)) + get_group_id(0);
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

inline void block_reduce_sum(local volatile int* sh_data)
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

kernel void accum_histogram_rgb(global const int *p_int_hist_data, global int *p_hist_data, int count)
{
    local int sh_data[64];
    int lid = get_local_id(0);
    int index = (get_group_id(0)*count);
    sh_data[get_local_id(0)] = 0;
    while (lid < count)
    {
        sh_data[get_local_id(0)] += p_int_hist_data[index + lid];
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
