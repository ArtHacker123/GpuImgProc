#include <iostream>

#include "OclReduceSum.h"

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

		std::string name;
		devices[0].getInfo<std::string>(CL_DEVICE_NAME, &name);
		std::cout << name << std::endl;

		int sum = 0;
        size_t dataSize = 640*480;
		Ocl::DataBuffer<int> buff(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, dataSize);
        int* pData = buff.map(queue, CL_TRUE, CL_MAP_WRITE, 0, dataSize);
		for (size_t i = 0; i < dataSize; i++)
		{
			pData[i] = 1;
			sum += 1;
		}
        buff.unmap(queue, pData);

		Ocl::ReduceSum rsum(context);

		for (int i = 0; i < 10; i++)
		{
			int resSum = rsum.process(queue, buff);
			printf("\n%d Reduce Sum: %d %s", i, resSum, (sum == resSum)?"Success":"Failed");
			if (resSum != sum)
			{
				break;
			}
		}
	}

	catch (cl::Error error)
	{
		std::cerr << "Error: " << error.what() << "(" << error.err() << ")" << std::endl;
	}

	return 0;
}
