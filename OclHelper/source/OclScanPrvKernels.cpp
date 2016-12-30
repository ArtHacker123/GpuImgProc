#include "OclScanPrv.h"

#define OCL_PROGRAM_SOURCE(s) #s

const char Ocl::ScanPrv::sSource[] = OCL_PROGRAM_SOURCE(

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

inline void block_scan(local volatile int* sh_data)
{
    const int i = get_local_id(0);
    const int j = i%WARP_SIZE;
    const int k = i / WARP_SIZE;

    warp_scan(j, i, sh_data);
    warp_scan(j, (int)(get_local_size(0) + i), sh_data);
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

    for (int p = 0; p < 2; p++)
    {
        int n = (i * 2) + p;
        sh_data[n] += sh_data[SH_MEM_SIZE + (n / WARP_SIZE)];
    }
}

kernel void prefix_sum(global int* p_data, int count)
{
    local int sh_data[SH_MEM_SIZE + WARP_SIZE];
    for (int i = 0; i < 2; i++)
    {
        int j = i + (2 * get_local_id(0));
        int k = i + (2 * get_global_id(0));
        sh_data[j] = (k < count) ? p_data[k] : 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    block_scan(sh_data);
    for (int i = 0; i < 2; i++)
    {
        int j = i + (2 * get_local_id(0));
        int k = i + (2 * get_global_id(0));
        if (k < count)
        {
            p_data[k] = sh_data[j];
        }
    }
}

kernel void gather_scan(global const int* p_data, int start, int count, global int* p_out_data)
{
    local int sh_data[SH_MEM_SIZE + WARP_SIZE];
    const int i = (get_local_id(0) * 2);
    const int j = (SH_MEM_SIZE*i) + (start - 1);
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

);
