#include "HistogramCoords.h"

#define OCL_PROGRAM_SOURCE(s) #s

const char HistogramCoords::source[] = OCL_PROGRAM_SOURCE(
kernel void histogram_coords(global read_only unsigned int* p_hist_data, unsigned int max_value, global float2* coord_r, global float2* coord_g, global float2* coord_b)
{
	int i = get_global_id(0);
	coord_r[i].x = -1.0f + (((float)i)/128.0f);
	coord_g[i].x = coord_b[i].x = coord_r[i].x;
	coord_r[i].y = -1.0f + (((float)p_hist_data[i]*2.0f)/(float)max_value);
	coord_g[i].y = -1.0f + (((float)p_hist_data[256+i]*2.0f)/(float)max_value);
	coord_b[i].y = -1.0f + (((float)p_hist_data[512+i]*2.0f)/(float)max_value);
};
);

HistogramCoords::HistogramCoords(cl::Context& ctxt, cl::CommandQueue& q)
	:mContext(ctxt),
	 mQueue(q)
{
	cl::Program::Sources source(1, std::make_pair(source, strlen(source)));
	mProgram = cl::Program(mContext, source);
	mProgram.build();

	mKernel = cl::Kernel(mProgram, "histogram_coords");
}

HistogramCoords::~HistogramCoords()
{
}

void HistogramCoords::compute(Ocl::DataBuffer<int> &rgbBins, unsigned int max_value, cl::BufferGL& rCoords, cl::BufferGL& gCoords, cl::BufferGL& bCoords)
{
	cl::Event event;
	std::vector<cl::Memory> gl_objs = { rCoords, gCoords, bCoords };

    mKernel.setArg(0, rgbBins.buffer());
	mKernel.setArg(1, max_value);
	mKernel.setArg(2, rCoords);
	mKernel.setArg(3, gCoords);
	mKernel.setArg(4, bCoords);

	mQueue.enqueueAcquireGLObjects(&gl_objs);
	mQueue.enqueueNDRangeKernel(mKernel, cl::NullRange, cl::NDRange(256), cl::NDRange(256), NULL, &event);
	event.wait();
	mQueue.enqueueReleaseGLObjects(&gl_objs);
}
