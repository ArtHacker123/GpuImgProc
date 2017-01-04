#include <iostream>
#include <cstdint>

#include "OclHoughLines.h"
#include "OclCompact.h"

void test_hough_lines(cl::Context& context, cl::CommandQueue& queue);

int main()
{
	try
	{
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.size() == 0)
		{
			std::cout << "Platform size 0\n";
			return -1;
		}

		cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
		cl::Context context(CL_DEVICE_TYPE_GPU, properties);

		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

		std::string name;
		devices[0].getInfo<std::string>(CL_DEVICE_NAME, &name);
		std::cout << name << std::endl;

		test_hough_lines(context, queue);
	}

	catch (cl::Error error)
	{
		std::cerr << "Error: " << error.what() << "(" << error.err() << ")" << std::endl;
	}

	return 0;
}

void test_hough_lines(cl::Context& context, cl::CommandQueue& queue)
{
	size_t width = 640;
	size_t height = 480;
    //size_t max_rho = 1+(size_t)(0.5*sqrt((double)((width*width)+(height*height))));
    Ocl::HoughLines houghLines(context, queue);
	cl::Image2D img(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), width, height);
	Ocl::DataBuffer<Ocl::HoughData> hdata(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 1000);
    
	size_t row_pitch = 0;
	size_t slice_pitch = 0;
	cl::size_t<3> img_orig;
	cl::size_t<3> img_region;

	img_region[0] = width;
	img_region[1] = height;
	img_region[2] = 1;

	float* pdata = (float *)queue.enqueueMapImage(img, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, img_orig, img_region, &row_pitch, &slice_pitch);
    row_pitch /= sizeof(float);
	for (size_t y = 0; y < height; y++)
	{
		float* pimg_data = (float*)&pdata[row_pitch*y];
		for (size_t x = 0; x < width; x++)
		{
			pimg_data[x] = 0.0;
		}
	}
    size_t x = 100;
    size_t y = 100;
    for (y = 100; y < 380; y++)
    {
        pdata[(row_pitch*y) + x] = 1.0f;
    }

    y = 100;
    for (x = 100; x < 540; x++)
    {
        pdata[(row_pitch*y) + x] = 1.0f;
    }

    /*for (size_t i = 0; i < (width*height) / 4; i++)
    {
        size_t x = rand()%width;
        size_t y = rand()%height;
        pdata[(row_pitch*y)+x] = 1.0f;
    }*/

	queue.enqueueUnmapMemObject(img, pdata);
	queue.finish();

    size_t outCount = 0;
    size_t time = houghLines.process(img, 100, hdata, outCount);
    printf("\nTime: %d ns", time);
    Ocl::HoughData* pHoughData = hdata.map(queue, CL_TRUE, CL_MAP_READ, 0, outCount);
    for (size_t i = 0; i < outCount; i++)
    {
        printf("\n%d: rho: %d, angle: %d, size: %d", i, pHoughData[i].rho, pHoughData[i].angle, pHoughData[i].strength);
    }
    hdata.unmap(queue, pHoughData);
}
