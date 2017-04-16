#include <iostream>

#include "OclCompact.h"
#include "OclUtils.h"

void test_image_compact(cl::Context& context, cl::CommandQueue& queue);

int main(int argc, char** argv)
{
    try
    {
        cl::Context context(CL_DEVICE_TYPE_GPU);
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        std::string name;
        devices[0].getInfo<std::string>(CL_DEVICE_NAME, &name);
        std::cout << name << std::endl;

        test_image_compact(context, queue);
    }

    catch (cl::Error error)
    {
        std::cerr << "Error: " << error.what() << "(" << error.err() << ")" << std::endl;
    }

    return 0;
}

void test_image_compact(cl::Context& context, cl::CommandQueue& queue)
{
    size_t width = 640;
    size_t height = 480;
    cl::Image2D image(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), width, height);

    size_t row_pitch = 0;
    size_t slice_pitch = 0;
    cl::size_t<3> img_orig;
    cl::size_t<3> img_region;

    img_region[0] = width;
    img_region[1] = height;
    img_region[2] = 1;

    float* pData = (float*)queue.enqueueMapImage(image, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, img_orig, img_region, &row_pitch, &slice_pitch);
    row_pitch /= sizeof(float);
    for (size_t y = 0; y < height; y++)
    {
        float* pImgData = (float*)&pData[row_pitch*y];
        for (size_t x = 0; x < width; x++)
        {
            pImgData[x] = 0.0f;
        }
    }

    for (size_t i = 0; i < 5; i++)
    {
        int x = rand() % width;
        int y = rand() % height;
        float* pImgData = (float*)&pData[row_pitch*y];
        pImgData[x] = 1.0f;
        printf("\n%d: Coords: (%d %d)", (int)i, x, y);
    }
    queue.enqueueUnmapMemObject(image, pData);

    Ocl::DataBuffer<cl_int2> outBuff(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 1000);

    Ocl::Compact compact(context);
    std::vector<cl::Event> events;
    Ocl::DataBuffer<cl_int> outCountBuff(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 1);

    for (size_t i = 0; i < 1; i++)
    {
        compact.process(queue, image, outBuff, 1.0f, outCountBuff, events);
        events.back().wait();
        size_t time = Ocl::kernelExecTime(queue, events.data(), events.size());

        cl_int outCount = 0;
        queue.enqueueReadBuffer(outCountBuff.buffer(), CL_TRUE, 0, sizeof(cl_int), &outCount);

        if (outCount > 0)
        {
            cl_int2* pData = outBuff.map(queue, CL_TRUE, CL_MAP_READ, 0, 100);
            for (int i = 0; i < outCount; i++)
            {
                printf("\n%d - (%d, %d)", i, pData[i].x, pData[i].y);
            }
            outBuff.unmap(queue, pData);
        }
        printf("\nTime: %d ns", time);
    }
}
