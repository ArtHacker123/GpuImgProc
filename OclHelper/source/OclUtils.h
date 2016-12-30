#pragma once

#include <CL/cl.hpp>

namespace Ocl
{

size_t localGroupSize(size_t size);

size_t kernelExecTime(cl::CommandQueue& queue, cl::Event& event);

size_t getWorkGroupSizeMultiple(const cl::CommandQueue& queue, const cl::Kernel& kernel);

}
