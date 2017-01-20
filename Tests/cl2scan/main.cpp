#include <iostream>
#include <sstream>

#include <CL/cl.hpp>
#include <CL/cl2.hpp>

#define OCL_PROGRAM_SOURCE(s) #s

const char sSource[] = OCL_PROGRAM_SOURCE(

kernel void child_scan(global int* pdata, int count)
{
    int i = get_global_id(0);
    int data = work_group_scan_inclusive_add((i < count)?pdata[i]:0);
    if (i < count) pdata[i] = data;
}

kernel void gather_scan(global const int* pdata, global int* tempData, int offset, int count)
{
    int i = offset+(get_local_size(0)*get_local_id(0))-1;
    tempData[get_local_id(0)] = work_group_scan_inclusive_add((i<count)?pdata[i]:0);
}

kernel void add_data(global int* pdata, global const int* tempData, int offset, int count)
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

kernel void scan(global int* pdata, global int* tempData, int count)
{
    clk_event_t event1;
    clk_event_t event2;
    const int BLK_SIZE = 256;
    const int BLK_SIZE2 = (BLK_SIZE*BLK_SIZE);
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(count, BLK_SIZE), 0, 0, &event1, ^{child_scan(pdata, count);});
    for (int i = BLK_SIZE; i < count; i += BLK_SIZE2)
    {
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(BLK_SIZE, BLK_SIZE), 1, &event1, &event2, ^{ gather_scan(pdata, tempData, i, count); });
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(BLK_SIZE2, BLK_SIZE), 1, &event2, &event1, ^{ add_data(pdata, tempData, i, count); });
    }
    release_event(event1);
}

);

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
        if (0 != version.find("OpenCL C 2.0"))
        {
            printf("\nOpenCL 2.0 required. Exiting...");
            return 0;
        }

        std::ostringstream options;
        options << "-cl-std=CL2.0";

        cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
        cl::Program program(context, source);
        program.build(options.str().c_str());

        int count = 640*480;
        cl_int* pIntData = (cl_int *)clSVMAlloc(context(), CL_MEM_READ_WRITE, 256*sizeof(cl_int), sizeof(cl_long));
        cl_int* pInpData = (cl_int *)clSVMAlloc(context(), CL_MEM_READ_WRITE, count*sizeof(cl_int), sizeof(cl_long));
        for (int i = 0; i < count; i++)
        {
            pInpData[i] = 1;
        }

        cl::Kernel kernel(program, "scan");
        clSetKernelArgSVMPointer(kernel(), 0, pInpData);
        clSetKernelArgSVMPointer(kernel(), 1, pIntData);
        kernel.setArg(2, count);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange, NULL, &event);
        event.wait();
        size_t time = (event.getProfilingInfo<CL_PROFILING_COMMAND_END>()-event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
        printf("\nTime: %d ns", time);

        clSVMFree(context(), pIntData);
        pIntData = 0;
        clSVMFree(context(), pInpData);
        pInpData = 0;
    }

    catch (cl::Error error)
    {
        std::cerr << "Error: " << error.what() << "(" << error.err() << ")" << std::endl;
    }

    return 0;
}
