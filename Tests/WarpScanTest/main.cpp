#include <iostream>

#include "OclScan.h"
#include "OclUtils.h"

int main(int argc, char** argv)
{
	try
	{
        cl::Context context(CL_DEVICE_TYPE_GPU);
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

		std::string name;
		devices[0].getInfo<std::string>(CL_DEVICE_NAME, &name);
		std::cout << name << std::endl;

        size_t dataSize = 640*480;
		Ocl::DataBuffer<cl_int> buff(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, dataSize);
		cl_int* pData = buff.map(queue, CL_TRUE, CL_MAP_WRITE, 0, dataSize);
		for (size_t i = 0; i < dataSize; i++)
		{
			pData[i] = 1;
		}
		buff.unmap(queue, pData);

		Ocl::Scan scan(context);
        std::vector<cl::Event> events;
		scan.process(queue, buff, events);
        events.back().wait();
        size_t time = Ocl::kernelExecTime(queue, events.data(), events.size());
		std::cout << "Time: " << time << " ns" << std::endl;
		
        bool flag = true;
		pData = buff.map(queue, CL_TRUE, CL_MAP_READ, 0, dataSize);;
		for (size_t i = 0; (i < dataSize) && flag; i++)
		{
			if (pData[i] != (int)(i + 1))
			{
				std::cerr << "Failed " << i << ", " << pData[i];
                flag = false;
			}
		}
		buff.unmap(queue, pData);

        std::cout << "Prefix Sum " << (flag ? "Success" : "Failed") << std::endl;
	}

	catch (cl::Error error)
	{
		std::cerr << "Error: " << error.what() << "(" << error.err() << ")" << std::endl;
	}

	return 0;
}
