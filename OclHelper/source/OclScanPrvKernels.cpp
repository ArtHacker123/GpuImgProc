#include "OclScanPrv.h"

#define OCL_PROGRAM_SOURCE(s) #s

const char Ocl::ScanPrv::sSource[] = OCL_PROGRAM_SOURCE(

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
