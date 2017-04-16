#include "OclUtils.h"

size_t Ocl::localGroupSize(size_t size)
{
    if ((size%16) == 0)
    {
        return 16;
    }

    if ((size%8) == 0)
    {
        return 8;
    }

    return 0;
}

size_t Ocl::kernelExecTime(const cl::CommandQueue& queue, const cl::Event& event)
{
    cl_command_queue_properties qProp;
    queue.getInfo<cl_command_queue_properties>(CL_QUEUE_PROPERTIES, &qProp);
    if (qProp & CL_QUEUE_PROFILING_ENABLE)
    {
        return (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    }
    return 0;
}

size_t Ocl::kernelExecTime(const cl::CommandQueue& queue, const cl::Event* event, size_t count)
{
    cl_command_queue_properties qProp;
    queue.getInfo<cl_command_queue_properties>(CL_QUEUE_PROPERTIES, &qProp);
    if (qProp & CL_QUEUE_PROFILING_ENABLE)
    {
        size_t time = 0;
        for (size_t i = 0; i < count; i++)
        {
            time += (event[i].getProfilingInfo<CL_PROFILING_COMMAND_END>() - event[i].getProfilingInfo<CL_PROFILING_COMMAND_START>());
        }
        return time;
    }
    return 0;
}

size_t Ocl::getWorkGroupSizeMultiple(const cl::CommandQueue& queue, const cl::Kernel& kernel)
{
    cl_device_id devId = 0;
    queue.getInfo<cl_device_id>(CL_QUEUE_DEVICE, &devId);

    size_t wgmSize = 0;
    cl::Device device(devId);
    kernel.getWorkGroupInfo<size_t>(device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &wgmSize);

    return wgmSize;
}
