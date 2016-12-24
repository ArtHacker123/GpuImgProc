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

size_t Ocl::kernelExecTime(cl::CommandQueue& queue, cl::Event& event)
{
    cl_command_queue_properties qProp;
    queue.getInfo<cl_command_queue_properties>(CL_QUEUE_PROPERTIES, &qProp);
    if (qProp & CL_QUEUE_PROFILING_ENABLE)
    {
        return (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    }
    return 0;
}
