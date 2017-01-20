#include "OglView.h"

#define OCL_PROGRAM_SOURCE(s) #s

const char OglView::sSource[] = OCL_PROGRAM_SOURCE(

typedef struct _CornerParam
{
    float rvalue;
    int cornerCount;
    int maxCornerCount;
} CornerParam;

inline void crossCoords(int i, int2 pos, int w, int h, global float2* coord)
{
    coord[i] = (float2)((float)(pos.x-4)/(float)w, (float)pos.y/(float)h);
    coord[i+1] = (float2)((float)(pos.x+4)/(float)w, (float)pos.y/(float)h);
    coord[i+2] = (float2)((float)pos.x/(float)w, (float)(pos.y-4)/(float)h);
    coord[i+3] = (float2)((float)pos.x/(float)w, (float)(pos.y+4)/(float)h);
}

kernel void extractCoords(global int2 *p_pos_corner, int count, global float2* coord, int width, int height)
{
    const int i = get_global_id(0);
    if (i > count) return;
    int2 corner = p_pos_corner[i];
    int2 pos = (int2)(corner.x-width, height-corner.y);
    crossCoords(4*i, pos, width, height, coord);
}

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
                int m = get_local_id(0)+(i*get_local_size(0));
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

    const int p = get_local_id(0)+1;
    const int q = get_local_id(1)+1;
    float ix = sh_img_data[q-1][p+1];
    ix += sh_img_data[q+1][p+1];
    ix -= sh_img_data[q-1][p-1];
    ix -= sh_img_data[q+1][p-1];
    ix += 2.0f*(sh_img_data[q][p+1]-sh_img_data[q][p-1]);
    float iy = sh_img_data[q-1][p-1];
    iy -= sh_img_data[q+1][p-1];
    iy += sh_img_data[q-1][p+1];
    iy -= sh_img_data[q+1][p+1];
    iy += 2.0f*(sh_img_data[q-1][p]-sh_img_data[q+1][p]);
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

kernel void nms(read_only image2d_t inpImg, write_only image2d_t outImg, float limit)
{
    local float shImgData[BLK_SIZE_Y+2][BLK_SIZE_X+2];
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_LINEAR;

    const int x = (get_group_id(0)*get_local_size(0))-1;
    const int y = (get_group_id(1)*get_local_size(1))-1;

    for (int j = 0; j < 2; j++)
    {
        int n = get_local_id(1)+(j*get_local_size(1));
        if (n < (BLK_SIZE_Y+2))
        {
            for (int i = 0; i < 2; i++)
            {
                int m = get_local_id(0)+(i*get_local_size(0));
                if (m < (BLK_SIZE_X+2))
                {
                    int2 coord = (int2)(x+m, y+n);
                    shImgData[n][m] = read_imagef(inpImg, sampler, coord).x;
                    coord.x += get_local_size(0);
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int p = get_local_id(0)+1;
    const int q = get_local_id(1)+1;
    float data = shImgData[q][p];
    if (shImgData[q][p-1] >= data) data = 0.0f;
    if (shImgData[q][p+1] >= data) data = 0.0f;
    if (shImgData[q-1][p-1] >= data) data = 0.0f;
    if (shImgData[q-1][p] >= data) data = 0.0f;
    if (shImgData[q-1][p+1] >= data) data = 0.0f;
    if (shImgData[q+1][p-1] >= data) data = 0.0f;
    if (shImgData[q+1][p] >= data) data = 0.0f;
    if (shImgData[q+1][p+1] >= data) data = 0.0f;
    if (data >= limit)
    {
        data = 1.0f;
    }
    write_imagef(outImg, (int2)(get_global_id(0), get_global_id(1)), data);
}

kernel void childScan(global int* pdata, int count)
{
    int i = get_global_id(0);
    int data = work_group_scan_inclusive_add((i < count)?pdata[i]:0);
    if (i < count) pdata[i] = data;
}

kernel void gatherScan(global read_only int* pdata, global int* tempData, int offset, int count)
{
    int i = offset+(get_local_size(0)*get_local_id(0))-1;
    tempData[get_local_id(0)] = work_group_scan_inclusive_add((i<count)?pdata[i]:0);
}

kernel void addData(global int* pdata, global read_only int* tempData, int offset, int count)
{
    local int shData;
    if (get_local_id(0) == 0)
    {
        shData = tempData[get_group_id(0)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int i = offset+get_global_id(0);
    if (i < count) pdata[i] += shData;
}

kernel void reduceImage(read_only image2d_t image, global int* reduceSum, const float threshold)
{
    int input = (read_imagef(image, (int2)(get_global_id(0), get_global_id(1))).x >= threshold)?1:0;
    int sum = work_group_reduce_add(input);
    if (get_local_id(0) == 0 && get_local_id(1) == 0)
    {
        int index = get_group_id(0)+(get_num_groups(0)*get_group_id(1));
        reduceSum[index] = sum;
    }
}

kernel void scan_image(read_only image2d_t image, global read_only int* offset, const float threshold, global int2* coords, global int* coord_count, const int max_size)
{
    local int sh_offs;
    if (get_local_id(0) == 0 && get_local_id(1) == 0)
    {
        int index = get_group_id(0)+(get_num_groups(0)*get_group_id(1))-1;
        sh_offs = (index < 0)?0:offset[index];
        if (get_group_id(0) == 0 && get_group_id(1) == 0)
        {
            int max_index = (get_num_groups(0)*get_num_groups(1))-1;
            int data_count = offset[max_index];
            *coord_count = (data_count < max_size)?data_count:max_size;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int input = (read_imagef(image, (int2)(get_global_id(0), get_global_id(1))).x >= threshold)?1:0;
    int output = work_group_scan_exclusive_add(input);
    if (input == 1)
    {
        int out_index = sh_offs+output;
        if (out_index < max_size)
        {
            coords[out_index] = (int2)(get_global_id(0), get_global_id(1));
        }
    }
}

kernel void harrisCorner(read_only image2d_t inpImg, read_write image2d_t tmpImg1, read_write image2d_t tmpImg2, global int2* pCorners, global int* pTempData, global CornerParam* pParam)
{
    size_t lSize[2];
    size_t gSize[2];

    lSize[0] = 32; lSize[1] = 8;
    gSize[0] = get_image_width(inpImg);
    gSize[1] = get_image_height(inpImg);
    int count = (int)((gSize[0]/lSize[0])*(gSize[1]/lSize[1]));

    clk_event_t event1;
    clk_event_t event2;
    float rvalue = pParam->rvalue;
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange_2D(gSize, lSize), 0, 0, &event1, ^{ gradient(inpImg, tmpImg1); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange_2D(gSize, lSize), 1, &event1, &event2, ^{ eigen(tmpImg1, tmpImg2); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange_2D(gSize, lSize), 1, &event2, &event1, ^{ nms(tmpImg2, tmpImg1, rvalue); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange_2D(gSize, lSize), 1, &event1, &event2, ^{ reduceImage(tmpImg1, pTempData, 1.0f); });
    
    const int BLK_SIZE = 256;
    const int BLK_SIZE2 = (BLK_SIZE*BLK_SIZE);
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange_1D(count, BLK_SIZE), 1, &event2, &event1, ^{ childScan(pTempData, count); });
    
    global int* pTempData1 = pTempData+count;
    int i = BLK_SIZE;
    for (int i = BLK_SIZE; i < count; i += BLK_SIZE2)
    {
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(BLK_SIZE, BLK_SIZE), 1, &event1, &event2, ^{ gatherScan(pTempData, pTempData1, i, count); });
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(BLK_SIZE2, BLK_SIZE), 1, &event2, &event1, ^{ addData(pTempData, pTempData1, i, count); });
    }

    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_2D(gSize, lSize), 1, &event1, &event2, ^{ scan_image(tmpImg1, pTempData, 1.0f, pCorners, (global int *)&pParam->cornerCount, pParam->maxCornerCount); });
    release_event(event2);
}

);
