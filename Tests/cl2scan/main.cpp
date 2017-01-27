#include <iostream>
#include <sstream>

#include <CL/cl2.hpp>

#define OCL_PROGRAM_SOURCE(s) #s

const char sSource[] = OCL_PROGRAM_SOURCE(

kernel void child_scan(global int4* pdata, int count)
{
    local int shData[1024];
    int id = get_global_id(0);
    ((int4 *)shData)[get_local_id(0)] = pdata[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    int i = get_local_id(0);
    shData[i] = work_group_scan_inclusive_add(shData[i]);
    barrier(CLK_LOCAL_MEM_FENCE);
    i += 255;
    shData[i] = work_group_scan_inclusive_add(shData[i]);
    barrier(CLK_LOCAL_MEM_FENCE);
    i += 255;
    shData[i] = work_group_scan_inclusive_add(shData[i]);
    barrier(CLK_LOCAL_MEM_FENCE);
    i += 255;
    shData[i] = work_group_scan_inclusive_add(shData[i]);
    barrier(CLK_LOCAL_MEM_FENCE);
    i += 255;
    int data = work_group_scan_inclusive_add((i<1024)?shData[i]:0);
    if (i < 1024) shData[i] = data;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (id < count) pdata[id] = ((int4 *)shData)[get_local_id(0)];
}

kernel void gather_scan(global const int* pdata, global int* tempData, int offset, int count)
{
    local int shData[1024];
    for (int i = 0; i < 4; i++)
    {
        int index = (4*get_local_id(0))+i;
        int j = offset+(1024*index)-1;
        shData[index] = (j < count)?pdata[j]:0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int i = get_local_id(0);
    shData[i] = work_group_scan_inclusive_add(shData[i]);
    barrier(CLK_LOCAL_MEM_FENCE);
    i += 255;
    shData[i] = work_group_scan_inclusive_add(shData[i]);
    barrier(CLK_LOCAL_MEM_FENCE);
    i += 255;
    shData[i] = work_group_scan_inclusive_add(shData[i]);
    barrier(CLK_LOCAL_MEM_FENCE);
    i += 255;
    shData[i] = work_group_scan_inclusive_add(shData[i]);
    barrier(CLK_LOCAL_MEM_FENCE);
    i += 255;
    int data = work_group_scan_inclusive_add((i<1024) ? shData[i] : 0);
    if (i < 1024) shData[i] = data;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < 4; i++)
    {
        int index = (4*get_local_id(0))+i;
        tempData[index] = shData[index];
    }
}

kernel void add_data(global int4* pdata, global const int* tempData, int offset, int count)
{
    local int shData;
    if (get_local_id(0) == 0)
    {
        shData = tempData[get_group_id(0)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int i = (256*get_group_id(0))+get_local_id(0)+(offset/4);
    int4 data = pdata[i];
    data += shData;
    pdata[i] = data;
}

kernel void scan(global int* pdata, global int* tempData, int count)
{
    clk_event_t event1;
    clk_event_t event2;
    const int BLK_SIZE = 256;
    const int BLK_SIZE2 = (BLK_SIZE*BLK_SIZE);
    int ncount = count/4;
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(ncount, BLK_SIZE), 0, 0, &event1, ^{child_scan(pdata, ncount);});
    for (int i = (4*BLK_SIZE); i < count; i += (16*BLK_SIZE2))
    {
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(BLK_SIZE, BLK_SIZE), 1, &event1, &event2, ^{ gather_scan(pdata, tempData, i, count); });
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D((4*BLK_SIZE2), BLK_SIZE), 1, &event2, &event1, ^{ add_data(pdata, tempData, i, count); });
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

        cl::Program program(context, std::string(sSource));
        program.build(options.str().c_str());

        int count = 640*480;
        cl::SVMAllocator<int, cl::SVMTraitFine<>> svmAlloc(context);
        std::vector<cl_int, cl::SVMAllocator<cl_int, cl::SVMTraitFine<>>> tempData(1024, 0, svmAlloc);
        std::vector<cl_int, cl::SVMAllocator<cl_int, cl::SVMTraitFine<>>> inpData(count, 1, svmAlloc);

        cl::Kernel kernel(program, "scan");
        kernel.setArg(0, inpData.data());
        kernel.setArg(1, tempData.data());
        kernel.setArg(2, count);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange, NULL, &event);
        event.wait();
        size_t time = (event.getProfilingInfo<CL_PROFILING_COMMAND_END>()-event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
        printf("\nTime: %d ns", time);

        bool flag = true;
        for (size_t i = 0; i < inpData.size() && flag; i++)
        {
            if (inpData[i] != (i+1))
            {
                printf("\nFailed %d %d", inpData[i], i+1);
                flag = false;
            }
        }
        printf("\nPrfix sum %s", (flag ? "success" : "failed"));
    }

    catch (cl::Error error)
    {
        std::cerr << "Error: " << error.what() << "(" << error.err() << ")" << std::endl;
    }

    return 0;
}
