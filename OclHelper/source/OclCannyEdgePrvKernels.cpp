#include "OclCannyEdgePrv.h"

#define OCL_PROGRAM_SOURCE(s) #s

using namespace Ocl;

const char CannyEdgePrv::sSource[] = OCL_PROGRAM_SOURCE(

kernel void gradient(read_only image2d_t inpImg, write_only image2d_t outImg)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_LINEAR;

    float ix = -read_imagef(inpImg, sampler, (int2)(x, y)).x;
    float iy = read_imagef(inpImg, sampler, (int2)(x, y)).x;
    iy += (2.0f*read_imagef(inpImg, sampler, (int2)(x+1, y)).x);
    ix += read_imagef(inpImg, sampler, (int2)(x+2, y)).x;
    iy += read_imagef(inpImg, sampler, (int2)(x+2, y)).x;

    ix -= (2.0f*read_imagef(inpImg, sampler, (int2)(x, y+1)).x);
    ix += (2.0f*read_imagef(inpImg, sampler, (int2)(x+2, y+1)).x);

    ix -= read_imagef(inpImg, sampler, (int2)(x, y+2)).x;
    iy -= read_imagef(inpImg, sampler, (int2)(x, y+2)).x;
    iy -= (2.0f*read_imagef(inpImg, sampler, (int2)(x+1, y+2)).x);
    ix += read_imagef(inpImg, sampler, (int2)(x+2, y+2)).x;
    iy -= read_imagef(inpImg, sampler, (int2)(x+2, y+2)).x;

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

    write_imagef(outImg, (int2)(x, y), (float4)(idata, theta, 0.0f, 0.0f));
}

kernel void gauss(read_only image2d_t inpImg, write_only image2d_t outImg)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_LINEAR;

    float idata = ((2.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x-2, y-2)).x);
    idata += ((4.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x-1, y-2)).x);
    idata += ((5.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x, y-2)).x);
    idata += ((4.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x+1, y-2)).x);
    idata += ((2.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x+2, y-2)).x);
    idata += ((4.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x-2, y-1)).x);
    idata += ((9.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x-1, y-1)).x);
    idata += ((12.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x, y-1)).x);
    idata += ((9.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x+1, y-1)).x);
    idata += ((4.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x+2, y-1)).x);
    idata += ((5.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x-2, y)).x);
    idata += ((12.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x-1, y)).x);
    idata += ((15.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x, y)).x);
    idata += ((12.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x+1, y)).x);
    idata += ((5.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x+2, y)).x);
    idata += ((4.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x-2, y+1)).x);
    idata += ((9.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x-1, y+1)).x);
    idata += ((12.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x, y+1)).x);
    idata += ((9.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x+1, y+1)).x);
    idata += ((4.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x+2, y+1)).x);
    idata += ((2.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x-2, y+2)).x);
    idata += ((4.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x-1, y+2)).x);
    idata += ((5.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x, y+2)).x);
    idata += ((4.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x+1, y+2)).x);
    idata += ((2.0f/159.0f)*read_imagef(inpImg, sampler, (int2)(x+2, y+2)).x);
    write_imagef(outImg, (int2)(x, y), idata);
}

//non max edge suppress
kernel void nmes(read_only image2d_t inpImg, write_only image2d_t outImg)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_LINEAR;

    float theta = read_imagef(inpImg, (int2)(x, y)).y;
    float idata = read_imagef(inpImg, sampler, (int2)(x, y)).x;

    if ((theta >= 22.5f && theta < 67.5f) || (theta >= 202.5f && theta < 247.5f))
    {
        if (idata <= read_imagef(inpImg, sampler, (int2)(x+1, y-1)).x || idata <= read_imagef(inpImg, sampler, (int2)(x-1, y+1)).x)
        {
            idata = 0.0f;
        }
    }
    else if ((theta >= 67.5f && theta < 112.5f) || (theta >= 247.5f && theta < 292.5f))
    {
        if (idata <= read_imagef(inpImg, sampler, (int2)(x, y-1)).x || idata <= read_imagef(inpImg, sampler, (int2)(x, y+1)).x)
        {
            idata = 0.0f;
        }
    }
    else if ((theta >= 112.5f && theta < 157.5f) || (theta >= 292.5f && theta < 337.5f))
    {
        if (idata <= read_imagef(inpImg, sampler, (int2)(x-1, y-1)).x || idata <= read_imagef(inpImg, sampler, (int2)(x+1, y+1)).x)
        {
            idata = 0.0f;
        }
    }
    else
    {
        if (idata <= read_imagef(inpImg, sampler, (int2)(x-1, y)).x || idata <= read_imagef(inpImg, sampler, (int2)(x+1, y)).x)
        {
            idata = 0.0f;
        }
    }
    write_imagef(outImg, (int2)(x, y), idata);
}

kernel void binary_threshold(read_only image2d_t inpImg, write_only image2d_t outImg, float minThresh, float maxThresh)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_LINEAR;
    float idata = read_imagef(inpImg, sampler, (int2)(x, y)).x;
    if (idata >= maxThresh)
    {
        idata = 1.0f;
    }
    else if (idata >= minThresh)
    {
        bool flag = true;
        if (maxThresh >= read_imagef(inpImg, sampler, (int2)(x-1, y)).x) flag = false;
        if (flag && maxThresh >= read_imagef(inpImg, sampler, (int2)(x-1, y-1)).x) flag = false;
        if (flag && maxThresh >= read_imagef(inpImg, sampler, (int2)(x, y-1)).x) flag = false;
        if (flag && maxThresh >= read_imagef(inpImg, sampler, (int2)(x+1, y-1)).x) flag = false;
        if (flag && maxThresh >= read_imagef(inpImg, sampler, (int2)(x+1, y)).x) flag = false;
        if (flag && maxThresh >= read_imagef(inpImg, sampler, (int2)(x+1, y+1)).x) flag = false;
        if (flag && maxThresh >= read_imagef(inpImg, sampler, (int2)(x, y+1)).x) flag = false;
        if (flag && maxThresh >= read_imagef(inpImg, sampler, (int2)(x-1, y+1)).x) flag = false;
        idata = (flag) ? 0.0f : 1.0f;
    }
    else
    {
        idata = 0.0f;
    }
    write_imagef(outImg, (int2)(x, y), idata);
}

);
