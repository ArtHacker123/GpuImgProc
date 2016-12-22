#include "HarrisCornerPrv.h"

#define OCL_PROGRAM_SOURCE(s) #s

using namespace Ocl;

const char HarrisCornerPrv::sSource[] = OCL_PROGRAM_SOURCE(

kernel void gradient(read_only image2d_t inpImg, write_only image2d_t outImg)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_LINEAR;
    local float sh_img_data[BLK_SIZE_Y+2][BLK_SIZE_X+2];

    const int x = (get_group_id(0)*get_local_size(0))-1;
    const int y = (get_group_id(1)*get_local_size(1))-1;

    for (int j = 0; j < 2; j++)
    {
        int n = get_local_id(1)+(j*get_local_size(1));
        if (n < (BLK_SIZE_Y+2))
        {
            for (int i = 0; i < 2; i++)
            {
                int m = get_local_id(0) + (i*get_local_size(0));
                if (m < (BLK_SIZE_X+2))
                {
                    int2 coord = (int2)(x+m, y+n);
                    sh_img_data[n][m] = read_imagef(inpImg, sampler, coord).x;
                    coord.x += get_local_size(0);
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int p = get_local_id(0) + 1;
    const int q = get_local_id(1) + 1;
    float ix = sh_img_data[q - 1][p + 1];
    ix += sh_img_data[q + 1][p + 1];
    ix -= sh_img_data[q - 1][p - 1];
    ix -= sh_img_data[q + 1][p - 1];
    ix += 2.0f*(sh_img_data[q][p + 1] - sh_img_data[q][p - 1]);
    float iy = sh_img_data[q - 1][p - 1];
    iy -= sh_img_data[q + 1][p - 1];
    iy += sh_img_data[q - 1][p + 1];
    iy -= sh_img_data[q + 1][p + 1];
    iy += 2.0f*(sh_img_data[q - 1][p] - sh_img_data[q + 1][p]);
    write_imagef(outImg, (int2)(get_global_id(0), get_global_id(1)), (float4)(ix, iy, 0.0f, 0.0f));
}

kernel void eigen(read_only image2d_t inpImg, write_only image2d_t mImg, global const float* p_coeffs)
{
    local float sh_coeffs[25];
    local float2 sh_img_data[BLK_SIZE_Y+4][BLK_SIZE_X+4];
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

    int index = (get_local_size(0)*get_local_id(1)) + get_local_id(0);
    if (index < 25)
    {
        sh_coeffs[index] = p_coeffs[index];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int x = (get_group_id(0)*get_local_size(0))-2;
    const int y = (get_group_id(1)*get_local_size(1))-2;

    for (int j = 0; j < 2; j++)
    {
        int n = get_local_id(1) + (j*get_local_size(1));
        if (n < (BLK_SIZE_Y+4))
        {
            for (int i = 0; i < 2; i++)
            {
                int m = get_local_id(0) + (i*get_local_size(0));
                if (m < (BLK_SIZE_X+4))
                {
                    int2 coord = (int2)(x + m, y + n);
                    sh_img_data[n][m] = read_imagef(inpImg, sampler, coord).xy;
                    coord.x += get_local_size(0);
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    index = 0;
    float ix2 = 0.0f;
    float iy2 = 0.0f;
    float ixiy = 0.0f;
    for (int i = -2; i <= 2; i++)
    {
        const int q = get_local_id(1) + i + 2;
        for (int j = -2; j <= 2; j++)
        {
            const int p = get_local_id(0) + j + 2;
            ix2 += (sh_coeffs[index] * sh_img_data[q][p].x*sh_img_data[q][p].x);
            iy2 += (sh_coeffs[index] * sh_img_data[q][p].y*sh_img_data[q][p].y);
            ixiy += (sh_coeffs[index] * sh_img_data[q][p].x*sh_img_data[q][p].y);
            ++index;
        }
    }
    float detA = (ix2*iy2) - (ixiy*ixiy);
    float traceA = (ix2 + iy2);
    float mValue = detA - (0.04*traceA*traceA);
    write_imagef(mImg, (int2)(get_global_id(0), get_global_id(1)), mValue);
}

kernel void suppress_non_max(read_only image2d_t inpImg, write_only image2d_t outImg, float limit)
{
    local float shImgData[BLK_SIZE_Y+2][BLK_SIZE_X+2];
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

    const int x = (get_group_id(0)*get_local_size(0))-1;
    const int y = (get_group_id(1)*get_local_size(1))-1;

    for (int j = 0; j < 2; j++)
    {
        int n = get_local_id(1) + (j*get_local_size(1));
        if (n < (BLK_SIZE_Y+2))
        {
            for (int i = 0; i < 2; i++)
            {
                int m = get_local_id(0) + (i*get_local_size(0));
                if (m < (BLK_SIZE_X+2))
                {
                    int2 coord = (int2)(x + m, y + n);
                    shImgData[n][m] = read_imagef(inpImg, sampler, coord).x;
                    coord.x += get_local_size(0);
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int p = get_local_id(0) + 1;
    const int q = get_local_id(1) + 1;
    float data = shImgData[q][p];
    if (shImgData[q][p - 1] >= data) data = 0.0f;
    if (shImgData[q][p + 1] >= data) data = 0.0f;
    if (shImgData[q - 1][p - 1] >= data) data = 0.0f;
    if (shImgData[q - 1][p] >= data) data = 0.0f;
    if (shImgData[q - 1][p + 1] >= data) data = 0.0f;
    if (shImgData[q + 1][p - 1] >= data) data = 0.0f;
    if (shImgData[q + 1][p] >= data) data = 0.0f;
    if (shImgData[q + 1][p + 1] >= data) data = 0.0f;
    if (data >= limit)
    {
        data = 1.0f;
    }
    write_imagef(outImg, (int2)(get_global_id(0), get_global_id(1)), data);
}

);
