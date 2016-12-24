#include "CannyEdgePrv.h"

#define OCL_PROGRAM_SOURCE(s) #s

using namespace Ocl;

const char CannyEdgePrv::sSource[] = OCL_PROGRAM_SOURCE(

inline void loadImageData(read_only image2d_t inpImg, local float* shImgData, const int p, const int q)
{
    const int xwidth = get_local_size(0)+(p*2);
    const int ywidth = get_local_size(1)+(q*2);
    const int x = (get_group_id(0)*get_local_size(0))-p;
    const int y = (get_group_id(1)*get_local_size(1))-q;
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_LINEAR;

    for (int n = get_local_id(1); n < ywidth; n += get_local_size(1))
    {
        for (int m = get_local_id(0); m < xwidth; m += get_local_size(0))
        {
            shImgData[(n*xwidth)+m] = read_imagef(inpImg, sampler, (int2)(x+m, y+n)).x;
        }
    }
}

inline void loadImageDataFloat2(read_only image2d_t inpImg, local float2* shImgData, const int p, const int q)
{
    const int xwidth = get_local_size(0)+(p*2);
    const int ywidth = get_local_size(1)+(q*2);
    const int x = (get_group_id(0)*get_local_size(0))-p;
    const int y = (get_group_id(1)*get_local_size(1))-q;
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_LINEAR;

    for (int n = get_local_id(1); n < ywidth; n += get_local_size(1))
    {
        for (int m = get_local_id(0); m < xwidth; m += get_local_size(0))
        {
            shImgData[(n*xwidth) + m] = read_imagef(inpImg, sampler, (int2)(x + m, y + n)).xy;
        }
    }
}

inline void loadCoeffs(global const float* pCoeffs, local float* shCoeffs, int count)
{
    int index = (get_local_size(0)*get_local_id(1)) + get_local_id(0);
    if (index < count)
    {
        shCoeffs[index] = pCoeffs[index];
    }
}

kernel void gradient(read_only image2d_t inpImg, write_only image2d_t outImg, local float* shImgData)
{
    loadImageData(inpImg, shImgData, 1, 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    const int xwidth = get_local_size(0)+2;
    int index = (xwidth*get_local_id(1))+get_local_id(0);

    float ix = -shImgData[index];
    float iy = shImgData[index];
    iy += (2.0f*shImgData[index+1]);
    ix += shImgData[index+2];
    iy += shImgData[index+2];
    index += xwidth;
    ix -= (2.0f*shImgData[index]);
    ix += (2.0f*shImgData[index+2]);
    index += xwidth;
    ix -= shImgData[index];
    iy -= shImgData[index];
    iy -= (2.0f*shImgData[index+1]);
    ix += shImgData[index+2];
    iy -= shImgData[index+2];

    float idata = sqrt((ix*ix) + (iy*iy));
    float theta = 180.0f*atan2pi(fabs(iy), fabs(ix));

    if (ix >= 0.0f && iy >= 0.0f)
        theta += 0.0f;
    else if (ix < 0.0f && iy >= 0.0f)
        theta = 180.0f - theta;
    else if (ix < 0.0f && iy < 0.0f)
        theta = 180.0f + theta;
    else
        theta = 360.0f - theta;

    /*if ((theta >= 22.5f && theta < 67.5f) || (theta >= 202.5f && theta < 247.5f))
        theta = 0.25f;
    else if ((theta >= 67.5f && theta < 112.5f) || (theta >= 247.5f && theta < 292.5f))
        theta = 0.5f;
    else if ((theta >= 112.5f && theta < 157.5f) || (theta >= 292.5f && theta < 337.5f))
        theta = 0.75f;
    else
        theta = 0.0f;*/

    write_imagef(outImg, (int2)(get_global_id(0), get_global_id(1)), (float4)(idata, theta, 0.0f, 0.0f));
}

kernel void gauss(read_only image2d_t inpImg, write_only image2d_t outImg, global const float* pCoeffs, local float* shImgData)
{
    local float shCoeffs[25];
    loadImageData(inpImg, shImgData, 2, 2);

    loadCoeffs(pCoeffs, shCoeffs, 25);
    barrier(CLK_LOCAL_MEM_FENCE);

    int index = 0;
    float idata = 0.0f;
    int xwidth = get_local_size(0)+4;
    for (int j = 0; j < 5; j++)
    {
        int y = (get_local_id(1)+j)*xwidth;
        for (int i = 0; i < 5; i++)
        {
            int offset = y+get_local_id(0)+i;
            idata += (shCoeffs[index++]*shImgData[offset]);
        }
    }
    write_imagef(outImg, (int2)(get_global_id(0), get_global_id(1)), idata);
}

kernel void non_max_edge_suppress(read_only image2d_t inpImg, write_only image2d_t outImg, local float2* shImgData)
{
    loadImageDataFloat2(inpImg, shImgData, 1, 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    const int p = get_local_id(0)+1;
    const int q = get_local_id(1)+1;
    const int xwidth = (get_local_size(0)+2);
    int index = (xwidth*q)+p;
    float idata = shImgData[index].x;
    float theta = shImgData[index].y;
    if ((theta >= 22.5f && theta < 67.5f) || (theta >= 202.5f && theta < 247.5f))
    {
        if (idata <= shImgData[(xwidth*(q-1))+p+1].x || idata <= shImgData[(xwidth*(q+1))+p-1].x)
        {
            idata = 0.0f;
        }
    }
    else if ((theta >= 67.5f && theta < 112.5f) || (theta >= 247.5f && theta < 292.5f))
    {
        if (idata <= shImgData[index-xwidth].x || idata <= shImgData[index+xwidth].x)
        {
            idata = 0.0f;
        }
    }
    else if ((theta >= 112.5f && theta < 157.5f) || (theta >= 292.5f && theta < 337.5f))
    {
        if (idata <= shImgData[(xwidth*(q-1))+p-1].x || idata <= shImgData[(xwidth*(q+1))+p+1].x)
        {
            idata = 0.0f;
        }
    }
    else
    {
        if (idata <= shImgData[index - 1].x || idata <= shImgData[index + 1].x)
        {
            idata = 0.0f;
        }
    }
    write_imagef(outImg, (int2)(get_global_id(0), get_global_id(1)), idata);
}

kernel void binary_threshold(read_only image2d_t inpImg, write_only image2d_t outImg, float minThresh, float maxThresh, local float* shImgData)
{
    loadImageData(inpImg, shImgData, 1, 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    const int p = get_local_id(0)+1;
    const int q = get_local_id(1)+1;
    const int xwidth = (get_local_size(0)+2);
    int index = (xwidth*q) + p;
    float idata = shImgData[index];
    if (idata >= maxThresh)
    {
        idata = 1.0f;
    }
    else if (idata >= minThresh)
    {
        int flag = 1;
        if (maxThresh >= shImgData[index-1]) flag = 0;
        if (maxThresh >= shImgData[index+1]) flag = 0;
        if (maxThresh >= shImgData[index-xwidth-1]) flag = 0;
        if (maxThresh >= shImgData[index-xwidth]) flag = 0;
        if (maxThresh >= shImgData[index-xwidth+1]) flag = 0;
        if (maxThresh >= shImgData[index+xwidth-1]) flag = 0;
        if (maxThresh >= shImgData[index+xwidth]) flag = 0;
        if (maxThresh >= shImgData[index+xwidth+1]) flag = 0;
        idata = (flag) ? 0.0f : 1.0f;
    }
    else
    {
        idata = 0.0f;
    }
    write_imagef(outImg, (int2)(get_global_id(0), get_global_id(1)), idata);
}

);
