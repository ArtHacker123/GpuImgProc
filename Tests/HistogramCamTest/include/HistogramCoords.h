#pragma once

#include "glew.h"
#include <GL/GL.h>

#include <CL/cl.hpp>
#include "DataBuffer.h"

class HistogramCoords
{
public:
	HistogramCoords(cl::Context& ctxt, cl::CommandQueue& queue);
	~HistogramCoords();

public:
	void compute(Ocl::DataBuffer<int>& rgbBins, unsigned int max_value, cl::BufferGL& rCoords, cl::BufferGL& gCoords, cl::BufferGL& bCoords);

private:
	cl::Context& mContext;
	cl::CommandQueue& mQueue;

	cl::Kernel mKernel;
	cl::Program mProgram;

	static const char source[];
};
