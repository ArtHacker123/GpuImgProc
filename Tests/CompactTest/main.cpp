#include <iostream>

#include "OclCompact.h"

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

		size_t dataSize = 640*480;
		cl::Buffer inpBuff(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, (size_t)(dataSize*sizeof(int)));
		int* pData = (int *)queue.enqueueMapBuffer(inpBuff, CL_TRUE, CL_MAP_WRITE, 0, dataSize*sizeof(int));
		memset(pData, 0, dataSize*sizeof(int));
		for (size_t i = 0; i < 5; i++)
		{
			int index = rand() % dataSize;
			pData[index] = 1;
			printf("\nindex = %d", index);
		}
		queue.enqueueUnmapMemObject(inpBuff, pData);

		cl::Buffer outBuff(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, (size_t)(100*sizeof(int)));

		OclCompact compact(context, queue);

		for (int i = 0; i < 10; i++)
		{
			size_t outSize = 0;
			compact.process(inpBuff, outBuff, outSize);

			int* pData = (int *)queue.enqueueMapBuffer(outBuff, CL_TRUE, CL_MAP_READ, 0, sizeof(int)*outSize);
			for (int i = 0; i < outSize; i++)
			{
				printf("\n%d %d", i, pData[i]);
			}
			queue.enqueueUnmapMemObject(outBuff, pData);
		}
	}

	catch (cl::Error error)
	{
		std::cerr << "Error: " << error.what() << "(" << error.err() << ")" << std::endl;
	}

	return 0;
}
