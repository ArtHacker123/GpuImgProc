#include "OclHoughLinesPrv.h"

#define OCL_PROGRAM_SOURCE(s) #s

using namespace Ocl;

const char HoughLinesPrv::sSource[] = OCL_PROGRAM_SOURCE(

kernel void non_max_suppress(read_only image2d_t inpImg, write_only image2d_t outImg, int threshold)
{
    local int shImgData[10][10];

    const int x = (get_group_id(0)*get_local_size(0)) - 1;
    const int y = (get_group_id(1)*get_local_size(1)) - 1;

    for (int j = 0; j < 2; j++)
    {
        int n = get_local_id(1) + (j*get_local_size(1));
        if (n < 10)
        {
            for (int i = 0; i < 2; i++)
            {
                int m = get_local_id(0) + (i*get_local_size(0));
                if (m < 10)
                {
                    int2 coord = (int2)(x + m, y + n);
                    if (coord.x < get_image_width(inpImg) && coord.y < get_image_height(inpImg))
                    {
                        shImgData[n][m] = read_imageui(inpImg, coord).x;
                    }
                    else
                    {
                        shImgData[n][m] = 0;
                    }
                    coord.x += get_local_size(0);
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int p = get_local_id(0) + 1;
    const int q = get_local_id(1) + 1;
    unsigned int data = shImgData[q][p];
    if (shImgData[q][p - 1] >= data) data = 0;
    if (shImgData[q][p + 1] >= data) data = 0;
    if (shImgData[q - 1][p - 1] >= data) data = 0;
    if (shImgData[q - 1][p] >= data) data = 0;
    if (shImgData[q - 1][p + 1] >= data) data = 0;
    if (shImgData[q + 1][p - 1] >= data) data = 0;
    if (shImgData[q + 1][p] >= data) data = 0;
    if (shImgData[q + 1][p + 1] >= data) data = 0;
    if (data < threshold)
    {
        data = 0;
    }
    write_imageui(outImg, (int2)(get_global_id(0), get_global_id(1)), data);
}

kernel void hough_line_transform(global const int2* p_coords, const int count, const int max_rho, write_only image2d_t outImg, local int* sh_rho)
{
    local float sh_sin_theta;
    local float sh_cos_theta;

    int i = get_local_id(0);
    if (i == 0)
    {
        float angle = (M_PI_F/180.0f)*(float)get_group_id(0);
        sh_sin_theta = sin(angle);
        sh_cos_theta = cos(angle);
    }

    while (i < max_rho)
    {
        sh_rho[i] = 0;
        i += get_local_size(0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    i = get_local_id(0);
    while (i < count)
    {
        int2 data = p_coords[i];
        i += get_local_size(0);
        int r = (int)ceil(((float)data.x*sh_cos_theta)+((float)data.y*sh_sin_theta));
        if (r >= 0)
        {
            atomic_inc(&sh_rho[r]);
        }
        //if (data.x == -220 && get_group_id(0) == 180) printf("\n%d %d %d, %f", data.x, data.y, sh_rho[r], sh_cos_theta);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    i = get_local_id(0);
    while (i < max_rho)
    {
        write_imageui(outImg, (int2)(i, get_group_id(0)), sh_rho[i]);
        i += get_local_size(0);
    }
}

);
