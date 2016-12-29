#include "OglHistogramPainterPrv.h"

using namespace Ogl;

#define OCL_PROGRAM_SOURCE(s) #s

const char HistogramPainterPrv::source[] = OCL_PROGRAM_SOURCE(
kernel void histogram_coords(global read_only unsigned int* p_hist_data, unsigned int max_value, global float2* coord)
{
    int i = get_global_id(0);
    coord[i].x = -1.0f + (((float)i)/128.0f);
    coord[i].y = -1.0f + (((float)p_hist_data[i]*2.0f)/(float)max_value);
};
);

HistogramPainterPrv::HistogramPainterPrv(cl::Context& ctxt, cl::CommandQueue& q)
    :mContext(ctxt),
     mQueue(q),
     mBuffer(GL_ARRAY_BUFFER, 512*sizeof(GLfloat), 0, GL_DYNAMIC_DRAW),
     mBufferGL(mContext, CL_MEM_READ_WRITE, mBuffer.buffer())
{
    cl::Program::Sources source(1, std::make_pair(source, strlen(source)));
    mProgram = cl::Program(mContext, source);
    mProgram.build();

    mKernel = cl::Kernel(mProgram, "histogram_coords");
}

HistogramPainterPrv::~HistogramPainterPrv()
{
}

void HistogramPainterPrv::setColor(GLfloat r, GLfloat g, GLfloat b)
{
    mPainter.SetColor(r, g, b, 1.0);
}

void HistogramPainterPrv::compute(const Ocl::DataBuffer<int> &hData, int maxValue)
{
    cl::Event event;
    std::vector<cl::Memory> gl_objs = { mBufferGL };

    mKernel.setArg(0, hData.buffer());
    mKernel.setArg(1, maxValue);
    mKernel.setArg(2, mBufferGL);

    mQueue.enqueueAcquireGLObjects(&gl_objs);
    mQueue.enqueueNDRangeKernel(mKernel, cl::NullRange, cl::NDRange(256), cl::NDRange(256), NULL, &event);
    event.wait();
    mQueue.enqueueReleaseGLObjects(&gl_objs);
}

void HistogramPainterPrv::draw(const Ocl::DataBuffer<int>& hData, int maxValue)
{
    compute(hData, maxValue);
    mPainter.draw(GL_LINE_STRIP, 0, (GLsizei)hData.count(), mBuffer);
}
