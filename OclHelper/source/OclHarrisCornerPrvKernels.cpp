#include "OclHarrisCornerPrv.h"

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

float gauss_smooth(local float sh_data1[BLK_SIZE_Y+4][BLK_SIZE_X+4], local float sh_data2[BLK_SIZE_Y+4][BLK_SIZE_X+4])
{
    const int m = get_local_id(0);
    const int n = get_local_id(1);
    float idata = ((2.0f/159.0f)*sh_data1[n][m]*sh_data2[n][m]);
    idata += ((4.0f/159.0f)*sh_data1[n][m+1]*sh_data2[n][m+1]);
    idata += ((5.0f/159.0f)*sh_data1[n][m+2]*sh_data2[n][m+2]);
    idata += ((4.0f/159.0f)*sh_data1[n][m+3]*sh_data2[n][m+3]);
    idata += ((2.0f/159.0f)*sh_data1[n][m+4]*sh_data2[n][m+4]);
    idata += ((4.0f/159.0f)*sh_data1[n+1][m]*sh_data2[n+1][m]);
    idata += ((9.0f/159.0f)*sh_data1[n+1][m+1]*sh_data2[n+1][m+1]);
    idata += ((12.0f/159.0f)*sh_data1[n+1][m+2]*sh_data2[n+1][m+2]);
    idata += ((9.0f/159.0f)*sh_data1[n+1][m+3]*sh_data2[n+1][m+3]);
    idata += ((4.0f/159.0f)*sh_data1[n+1][m+4]*sh_data2[n+1][m+4]);
    idata += ((5.0f/159.0f)*sh_data1[n+2][m]*sh_data2[n+2][m]);
    idata += ((12.0f/159.0f)*sh_data1[n+2][m+1]*sh_data2[n+2][m+1]);
    idata += ((15.0f/159.0f)*sh_data1[n+2][m+2]*sh_data2[n+2][m+2]);
    idata += ((12.0f/159.0f)*sh_data1[n+2][m+3]*sh_data2[n+2][m+3]);
    idata += ((5.0f/159.0f)*sh_data1[n+2][m+4]*sh_data2[n+2][m+4]);
    idata += ((4.0f/159.0f)*sh_data1[n+3][m]*sh_data2[n+3][m]);
    idata += ((9.0f/159.0f)*sh_data1[n+3][m+1]*sh_data2[n+3][m+1]);
    idata += ((12.0f/159.0f)*sh_data1[n+3][m+2]*sh_data2[n+3][m+2]);
    idata += ((9.0f/159.0f)*sh_data1[n+3][m+3]*sh_data2[n+3][m+3]);
    idata += ((4.0f/159.0f)*sh_data1[n+3][m+4]*sh_data2[n+3][m+4]);
    idata += ((2.0f/159.0f)*sh_data1[n+4][m]*sh_data2[n+4][m]);
    idata += ((4.0f/159.0f)*sh_data1[n+4][m+1]*sh_data2[n+4][m+1]);
    idata += ((5.0f/159.0f)*sh_data1[n+4][m+2]*sh_data2[n+4][m+2]);
    idata += ((4.0f/159.0f)*sh_data1[n+4][m+3]*sh_data2[n+4][m+3]);
    idata += ((2.0f/159.0f)*sh_data1[n+4][m+4]*sh_data2[n+4][m+4]);
    return idata;
}

kernel void eigen(read_only image2d_t inpImg, write_only image2d_t outImg)
{
    local float sh_img_data_x[BLK_SIZE_Y+4][BLK_SIZE_X+4];
    local float sh_img_data_y[BLK_SIZE_Y+4][BLK_SIZE_X+4];
    
    {
        const int x = (get_group_id(0)*get_local_size(0)) - 2;
        const int y = (get_group_id(1)*get_local_size(1)) - 2;
        const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_LINEAR;

        for (int j = 0; j < 2; j++)
        {
            int n = get_local_id(1) + (j*get_local_size(1));
            if (n < (BLK_SIZE_Y + 4))
            {
                for (int i = 0; i < 2; i++)
                {
                    int m = get_local_id(0) + (i*get_local_size(0));
                    if (m < (BLK_SIZE_X + 4))
                    {
                        int2 coord = (int2)(x + m, y + n);
                        float2 idata = read_imagef(inpImg, sampler, coord).xy;
                        sh_img_data_x[n][m] = idata.x;
                        sh_img_data_y[n][m] = idata.y;
                        coord.x += get_local_size(0);
                    }
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float ix2 = gauss_smooth(sh_img_data_x, sh_img_data_x);
    float iy2 = gauss_smooth(sh_img_data_y, sh_img_data_y);
    float ixiy = gauss_smooth(sh_img_data_x, sh_img_data_y);

    float detA = (ix2*iy2) - (ixiy*ixiy);
    float traceA = (ix2 + iy2);
    float mValue = detA - (0.04*traceA*traceA);
    write_imagef(outImg, (int2)(get_global_id(0), get_global_id(1)), mValue);
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
