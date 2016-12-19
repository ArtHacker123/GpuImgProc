#include <iostream>

#include "OclHistogram.h"
#include "HistogramRGB.h"

void test_histogram(cl::Context& context, cl::CommandQueue& queue);
void test_histogram_rgb(cl::Context& context, cl::CommandQueue& queue);

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

		//test_histogram(context, queue);
		test_histogram_rgb(context, queue);
	}

	catch (cl::Error error)
	{
		std::cerr << "Error: " << error.what() << "(" << error.err() << ")" << std::endl;
	}

	return 0;
}

void test_histogram(cl::Context& context, cl::CommandQueue& queue)
{
	size_t width = 1920;
	size_t height = 1080;
	cl::Image2D img(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_UNSIGNED_INT8), width, height);
	cl::Buffer hist_data(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, (size_t)(256*sizeof(int)));
	
	size_t row_pitch = 0;
	size_t slice_pitch = 0;
	cl::size_t<3> img_orig;
	cl::size_t<3> img_region;

	img_region[0] = width;
	img_region[1] = height;
	img_region[2] = 1;
	uint8_t* pdata = (uint8_t *)queue.enqueueMapImage(img, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, img_orig, img_region, &row_pitch, &slice_pitch);
	for (size_t y = 0; y < height; y++)
	{
		uint8_t* pimg_data = pdata + (row_pitch*y);
		for (size_t x = 0; x < width; x++)
		{
			pimg_data[x] = rand()/256;
		}
	}
	queue.enqueueUnmapMemObject(img, pdata);
	queue.finish();

	OclHistogram histogram(context, queue);
	for (size_t i = 0; i < 10; i++)
	{
		histogram.compute(img, hist_data);
	}

	int* img_data = (int *)queue.enqueueMapBuffer(hist_data, CL_TRUE, CL_MAP_READ, 0, 256*sizeof(int));
	int sum = 0;
	for (size_t i = 0; i < 256; i++)
	{
		sum += img_data[i];
	}
	queue.enqueueUnmapMemObject(hist_data, img_data);

	printf("\n%s", (sum == (width*height)) ? "Success" : "Failed");
}

void test_histogram_rgb(cl::Context& context, cl::CommandQueue& queue)
{
	size_t width = 1920;
	size_t height = 1080;
	cl::Image2D img(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), width, height);
	Ocl::DataBuffer<int> hist_data(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 3*256);

	size_t row_pitch = 0;
	size_t slice_pitch = 0;
	cl::size_t<3> img_orig;
	cl::size_t<3> img_region;

	img_region[0] = width;
	img_region[1] = height;
	img_region[2] = 1;

	uint8_t* pdata = (uint8_t *)queue.enqueueMapImage(img, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, img_orig, img_region, &row_pitch, &slice_pitch);
	for (size_t y = 0; y < height; y++)
	{
		uint32_t* pimg_data = (uint32_t*)&pdata[row_pitch*y];
		for (size_t x = 0; x < width; x++)
		{
            uint8_t red = rand() % 256;
            uint8_t green = rand() % 256;
            uint8_t blue = rand() % 256;
			pimg_data[x] = (blue<<16)|(green<<8)|red;
		}
	}
	queue.enqueueUnmapMemObject(img, pdata);
	queue.finish();

	Ocl::HistogramRGB histogram(context, queue);
	for (size_t i = 0; i < 10; i++)
	{
		size_t time = histogram.compute(img, hist_data);
        printf("\nTime: %f", (float)(((double)time)/1000000.0));
	}

	size_t start = 0;
	int sum;
	bool flag = true;
    int* img_data = hist_data.map(queue, CL_TRUE, CL_MAP_READ, 0, 256*3);
	for (size_t j = 0; j < 3; j++)
	{
		sum = 0;
		for (size_t i = start; i < (start+256); i++)
		{
			sum += img_data[i];
		}
		start += 256;
		flag = flag && (sum == (width*height));
	}
    hist_data.unmap(queue, img_data);

	printf("\n%s", flag ? "Success" : "Failed");
}

