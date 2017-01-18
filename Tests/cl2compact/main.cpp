#include <iostream>
#include <sstream>

#include <CL/cl.hpp>
#include <CL/cl2.hpp>

#define OCL_PROGRAM_SOURCE(s) #s

const char sSource[] = OCL_PROGRAM_SOURCE(

kernel void child_scan(global int* pdata, size_t count)
{
    int i = get_global_id(0);
    int data = work_group_scan_inclusive_add((i < count)?pdata[i]:0);
    if (i < count) pdata[i] = data;
}

kernel void gather_scan(global read_only int* pdata, global int* tempData, size_t offset, size_t count)
{
    int i = offset+(get_local_size(0)*get_local_id(0))-1;
    tempData[get_local_id(0)] = work_group_scan_inclusive_add((i<count)?pdata[i]:0);
}

kernel void add_data(global int* pdata, global read_only int* tempData, size_t offset, size_t count)
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

kernel void reduce_image(read_only image2d_t image, global int* reduceSum, const float threshold)
{
    int input = (read_imagef(image, (int2)(get_global_id(0), get_global_id(1))).x >= threshold)?1:0;
    int sum = work_group_reduce_add(input);
    if (get_local_id(0) == 0 && get_local_id(1) == 0)
    {
        size_t index = get_group_id(0)+(get_num_groups(0)*get_group_id(1));
        reduceSum[index] = sum;
    }
}

kernel void scan_image(read_only image2d_t image, global read_only int* offset, const float threshold, global int2* coords, global int* coord_count, const size_t max_size)
{
    local size_t sh_offs;
    if (get_local_id(0) == 0 && get_local_id(1) == 0)
    {
        size_t index = get_group_id(0)+(get_num_groups(0)*get_group_id(1))-1;
        sh_offs = (index < 0)?0:offset[index];
        if (get_group_id(0) == 0 && get_group_id(1) == 0)
        {
            size_t max_index = (get_num_groups(0)*get_num_groups(1))-1;
            size_t data_count = offset[max_index];
            *coord_count = (data_count < max_size)?data_count:max_size;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int input = (read_imagef(image, (int2)(get_global_id(0), get_global_id(1))).x >= threshold)?1:0;
    int output = work_group_scan_exclusive_add(input);
    if (input == 1)
    {
        size_t out_index = sh_offs+output;
        if (out_index < max_size)
        {
            coords[out_index] = (int2)(get_global_id(0), get_global_id(1));
        }
    }
}

kernel void compact(read_only image2d_t image, global int* tempData, const float threshold, global int2* coords, global int* coord_count, const int max_size)
{
    size_t lSize[2];
    size_t gSize[2];

    lSize[0] = 32; lSize[1] = 8;
    gSize[0] = get_image_width(image);
    gSize[1] = get_image_height(image);
    size_t count = (gSize[0]/lSize[0])*(gSize[1]/lSize[1]);

    clk_event_t event1;
    clk_event_t event2;
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_2D(gSize, lSize), 0, 0, &event1, ^{reduce_image(image, tempData, threshold);});

    const int BLK_SIZE = 256;
    const int BLK_SIZE2 = (BLK_SIZE*BLK_SIZE);
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(count, BLK_SIZE), 1, &event1, &event2, ^{child_scan(tempData, count);});

    global int* tempData1 = tempData+count;
    for (size_t i = BLK_SIZE; i < count; i += BLK_SIZE2)
    {
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(BLK_SIZE, BLK_SIZE), 1, &event2, &event1, ^{gather_scan(tempData, tempData1, i, count);});
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(BLK_SIZE2, BLK_SIZE), 1, &event1, &event2, ^{add_data(tempData, tempData1, i, count);});
    }

    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_2D(gSize, lSize), 1, &event2, &event1, ^{scan_image(image, tempData, threshold, coords, coord_count, max_size);});
    release_event(event1);
}

);

void test_compact(const cl::Context& context, const cl::CommandQueue& queue);

int main(int argc, char** argv)
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

        int err = 0;
        cl_queue_properties qprop[] = { CL_QUEUE_PROPERTIES, (cl_command_queue_properties)(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_ON_DEVICE|CL_QUEUE_ON_DEVICE_DEFAULT|CL_QUEUE_PROFILING_ENABLE), 0 };
        cl_command_queue dev_q = clCreateCommandQueueWithProperties(context(), devices[0](), qprop, &err);

		std::string name, version;
		devices[0].getInfo<std::string>(CL_DEVICE_NAME, &name);
        devices[0].getInfo<std::string>(CL_DEVICE_OPENCL_C_VERSION, &version);
		std::cout << name << ", " << version << std::endl;

        test_compact(context, queue);
	}

	catch (cl::Error error)
	{
		std::cerr << "Error: " << error.what() << "(" << error.err() << ")" << std::endl;
	}

	return 0;
}

void test_compact(const cl::Context& context, const cl::CommandQueue& queue)
{
    size_t width = 1920;
    size_t height = 1080;
    cl::Image2D image(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), width, height);
    
    size_t row_pitch = 0;
    size_t slice_pitch = 0;
    cl::size_t<3> img_orig;
    cl::size_t<3> img_region;

    img_region[0] = width;
    img_region[1] = height;
    img_region[2] = 1;

    float* pData = (float*)queue.enqueueMapImage(image, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, img_orig, img_region, &row_pitch, &slice_pitch);
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
    
    size_t tempCount = (width/32)*(height/8);
    tempCount += 256;
    cl::Buffer tempBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int)*tempCount);

    size_t maxCoordCount = 1000;
    cl::Buffer coordBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int2)*maxCoordCount);
    cl::Buffer coordCount(context, CL_MEM_READ_WRITE, sizeof(cl_int));

    std::ostringstream options;
    options << "-cl-std=CL2.0";

    cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
    cl::Program program(context, source);
    program.build(options.str().c_str());

    cl::Kernel kernel(program, "compact");
    kernel.setArg(0, image);
    kernel.setArg(1, tempBuffer);
    kernel.setArg(2, 1.0f);
    kernel.setArg(3, coordBuffer);
    kernel.setArg(4, coordCount);
    kernel.setArg(5, (int)maxCoordCount);
    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1, 1), cl::NullRange, NULL, &event);
    event.wait();
    size_t time = (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    printf("\nTime: %d ns", time);

    cl_int countCoords = 0;
    queue.enqueueReadBuffer(coordCount, CL_TRUE, 0, sizeof(cl_int), &countCoords);
    cl_int2* pCoordData = (cl_int2 *)queue.enqueueMapBuffer(coordBuffer, CL_TRUE, CL_MAP_READ, 0, sizeof(cl_int2)*countCoords);
    for (size_t i = 0; i < (size_t)countCoords; i++)
    {
        printf("\n%d: (%d %d)", i, pCoordData[i].x, pCoordData[i].y);
    }
    queue.enqueueUnmapMemObject(coordBuffer, pCoordData);
}
