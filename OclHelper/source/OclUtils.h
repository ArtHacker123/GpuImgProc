#pragma once

#include <CL/cl.hpp>

namespace Ocl
{

size_t localGroupSize(size_t size);

size_t kernelExecTime(cl::CommandQueue& queue, cl::Event& event);

}
