#include "OclCannyEdgePrv.h"

#define OCL_PROGRAM_SOURCE(s) #s

using namespace Ocl;

const char CannyEdgePrv::sSource[] = OCL_PROGRAM_SOURCE(

kernel void gradient(read_only image2d_t inpImg, write_only image2d_t outImg)
{
    local float sh_img_data[10][18];
    {
        const int x = (16*get_group_id(0))-1;
        const int y = (8*get_group_id(1))-1;
        const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_LINEAR;

        int n = get_local_id(1);
        do
        {
            int m = get_local_id(0);
            do
            {
                sh_img_data[n][m] = read_imagef(inpImg, sampler, (int2)(x+m, y+n)).x;
                m += get_local_size(0);
            } while (m < 18);
            n += get_local_size(1);
        } while (n < 10);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int m = get_local_id(0);
    int n = get_local_id(1);

    float ix = -sh_img_data[n][m];
    float iy = sh_img_data[n][m];
    iy += (2.0f*sh_img_data[n][m+1]);
    ix += sh_img_data[n][m+2];
    iy += sh_img_data[n][m+2];
    
    ++n;
    ix -= (2.0f*sh_img_data[n][m]);
    ix += (2.0f*sh_img_data[n][m+2]);
    
    ++n;
    ix -= sh_img_data[n][m];
    iy -= sh_img_data[n][m];
    iy -= (2.0f*sh_img_data[n][m+1]);
    ix += sh_img_data[n][m+2];
    iy -= sh_img_data[n][m+2];

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

    write_imagef(outImg, (int2)(get_global_id(0), get_global_id(1)), (float4)(idata, theta, 0.0f, 0.0f));
}

kernel void gauss(read_only image2d_t inpImg, write_only image2d_t outImg)
{
    local float sh_img_data[12][20];
    {
        const int x = (16*get_group_id(0))-2;
        const int y = (8*get_group_id(1))-2;
        const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_LINEAR;

        int n = get_local_id(1);
        do
        {
            int m = get_local_id(0);
            do
            {
                sh_img_data[n][m] = read_imagef(inpImg, sampler, (int2)(x+m, y+n)).x;
                m += get_local_size(0);
            } while (m < 20);
            n += get_local_size(1);
        } while (n < 12);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int m = get_local_id(0);
    const int n = get_local_id(1);
    float idata = ((2.0f/159.0f)*sh_img_data[n][m]);
    idata += ((4.0f/159.0f)*sh_img_data[n][m+1]);
    idata += ((5.0f/159.0f)*sh_img_data[n][m+2]);
    idata += ((4.0f/159.0f)*sh_img_data[n][m+3]);
    idata += ((2.0f/159.0f)*sh_img_data[n][m+4]);
    idata += ((4.0f/159.0f)*sh_img_data[n+1][m]);
    idata += ((9.0f/159.0f)*sh_img_data[n+1][m+1]);
    idata += ((12.0f/159.0f)*sh_img_data[n+1][m+2]);
    idata += ((9.0f/159.0f)*sh_img_data[n+1][m+3]);
    idata += ((4.0f/159.0f)*sh_img_data[n+1][m+4]);
    idata += ((5.0f/159.0f)*sh_img_data[n+2][m]);
    idata += ((12.0f/159.0f)*sh_img_data[n+2][m+1]);
    idata += ((15.0f/159.0f)*sh_img_data[n+2][m+2]);
    idata += ((12.0f/159.0f)*sh_img_data[n+2][m+3]);
    idata += ((5.0f/159.0f)*sh_img_data[n+2][m+4]);
    idata += ((4.0f/159.0f)*sh_img_data[n+3][m]);
    idata += ((9.0f/159.0f)*sh_img_data[n+3][m+1]);
    idata += ((12.0f/159.0f)*sh_img_data[n+3][m+2]);
    idata += ((9.0f/159.0f)*sh_img_data[n+3][m+3]);
    idata += ((4.0f/159.0f)*sh_img_data[n+3][m+4]);
    idata += ((2.0f/159.0f)*sh_img_data[n+4][m]);
    idata += ((4.0f/159.0f)*sh_img_data[n+4][m+1]);
    idata += ((5.0f/159.0f)*sh_img_data[n+4][m+2]);
    idata += ((4.0f/159.0f)*sh_img_data[n+4][m+3]);
    idata += ((2.0f/159.0f)*sh_img_data[n+4][m+4]);
    write_imagef(outImg, (int2)(get_global_id(0), get_global_id(1)), idata);
}

kernel void non_max_edge_suppress(read_only image2d_t inpImg, write_only image2d_t outImg)
{
    local float sh_img_data[10][18];
    {
        const int x = (16*get_group_id(0))-1;
        const int y = (8*get_group_id(1))-1;
        const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_LINEAR;

        int n = get_local_id(1);
        do
        {
            int m = get_local_id(0);
            do
            {
                sh_img_data[n][m] = read_imagef(inpImg, sampler, (int2)(x+m, y+n)).x;
                m += get_local_size(0);
            } while (m < 18);
            n += get_local_size(1);
        } while (n < 10);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int p = get_local_id(0)+1;
    const int q = get_local_id(1)+1;
    float idata = sh_img_data[q][p];
    float theta = read_imagef(inpImg, (int2)(get_global_id(0), get_global_id(1))).y;
    if ((theta >= 22.5f && theta < 67.5f) || (theta >= 202.5f && theta < 247.5f))
    {
        if (idata <= sh_img_data[q-1][p+1] || idata <= sh_img_data[q+1][p-1])
        {
            idata = 0.0f;
        }
    }
    else if ((theta >= 67.5f && theta < 112.5f) || (theta >= 247.5f && theta < 292.5f))
    {
        if (idata <= sh_img_data[q-1][p] || idata <= sh_img_data[q+1][p])
        {
            idata = 0.0f;
        }
    }
    else if ((theta >= 112.5f && theta < 157.5f) || (theta >= 292.5f && theta < 337.5f))
    {
        if (idata <= sh_img_data[q-1][p-1] || idata <= sh_img_data[q+1][p+1])
        {
            idata = 0.0f;
        }
    }
    else
    {
        if (idata <= sh_img_data[q][p-1] || idata <= sh_img_data[q][p+1])
        {
            idata = 0.0f;
        }
    }
    write_imagef(outImg, (int2)(get_global_id(0), get_global_id(1)), idata);
}

kernel void binary_threshold(read_only image2d_t inpImg, write_only image2d_t outImg, float minThresh, float maxThresh)
{
    local float sh_img_data[10][18];
    {
        const int x = (16*get_group_id(0))-1;
        const int y = (8*get_group_id(1))-1;
        const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_LINEAR;

        int n = get_local_id(1);
        do
        {
            int m = get_local_id(0);
            do
            {
                sh_img_data[n][m] = read_imagef(inpImg, sampler, (int2)(x+m, y+n)).x;
                m += get_local_size(0);
            } while (m < 18);
            n += get_local_size(1);
        } while (n < 10);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int p = get_local_id(0)+1;
    const int q = get_local_id(1)+1;
    float idata = sh_img_data[q][p];
    if (idata >= maxThresh)
    {
        idata = 1.0f;
    }
    else if (idata >= minThresh)
    {
        int flag = 1;
        if (maxThresh >= sh_img_data[q][p-1]) flag = 0;
        if (maxThresh >= sh_img_data[q][p+1]) flag = 0;
        if (maxThresh >= sh_img_data[q-1][p-1]) flag = 0;
        if (maxThresh >= sh_img_data[q-1][p]) flag = 0;
        if (maxThresh >= sh_img_data[q-1][p+1]) flag = 0;
        if (maxThresh >= sh_img_data[q+1][p-1]) flag = 0;
        if (maxThresh >= sh_img_data[q+1][p]) flag = 0;
        if (maxThresh >= sh_img_data[q+1][p+1]) flag = 0;
        idata = (flag) ? 0.0f : 1.0f;
    }
    else
    {
        idata = 0.0f;
    }
    write_imagef(outImg, (int2)(get_global_id(0), get_global_id(1)), idata);
}

);
