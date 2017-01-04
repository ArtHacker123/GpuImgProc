#include "OclHoughLinesPrv.h"

#define OCL_PROGRAM_SOURCE(s) #s

using namespace Ocl;

const char HoughLinesPrv::sSource[] = OCL_PROGRAM_SOURCE(

kernel void convert_coords(global const int2* p_inp, global int2*p_out, const int count, const int width, const int height)
{
    if (get_global_id(0) >= count) return;
    int2 data = p_inp[get_global_id(0)];
    data.x -= width;
    data.y = height-data.y;
    p_out[get_global_id(0)] = data;
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
