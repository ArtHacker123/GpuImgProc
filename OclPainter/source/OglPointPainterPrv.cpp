#include "OglPointPainterPrv.h"

#define OCL_PROGRAM_SOURCE(s) #s

using namespace Ogl;

const char PointPainterPrv::sSource[] = OCL_PROGRAM_SOURCE(
inline void cross_coords(int i, int2 pos, int w, int h, global float2* coord)
{
    coord[i] = (float2)((float)(pos.x-4)/(float)w, (float)pos.y/(float)h);
    coord[i+1] = (float2)((float)(pos.x+4)/(float)w, (float)pos.y/(float)h);
    coord[i+2] = (float2)((float)pos.x/(float)w, (float)(pos.y-4)/(float)h);
    coord[i+3] = (float2)((float)pos.x/(float)w, (float)(pos.y+4)/(float)h);
}

kernel void extract_coords(global int2 *p_pos_corner, int count, global float2* coord, int width, int height)
{
    const int i = get_global_id(0);
    if (i > count) return;
    int2 corner = p_pos_corner[i];
    int2 pos = (int2)(corner.x-width, height-corner.y);
    cross_coords(4*i, pos, width, height, coord);
}
);

PointPainterPrv::PointPainterPrv(const cl::Context& ctxt, size_t maxPoints)
    :mMaxPoints(maxPoints),
	 mContext(ctxt),
	 mPointBuff(GL_ARRAY_BUFFER, (8*mMaxPoints*sizeof(GLfloat)), 0, GL_DYNAMIC_DRAW)
{
    try
    {
        cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
        mPgm = cl::Program(mContext, source);
        mPgm.build();
        mKernel = cl::Kernel(mPgm, "extract_coords");
        mPainter.SetColor(1.0f, 0.0f, 0.0f, 1.0f);
    }

    catch (cl::Error error)
    {
        printf("\nFailed: %s", error.what());
        exit(0);
    }
}

PointPainterPrv::~PointPainterPrv()
{
}

void PointPainterPrv::draw(const cl::CommandQueue& queue, Ocl::DataBuffer<cl_int2>& points, size_t count, size_t width, size_t height)
{
    if (count <= 0)
    {
        return;
    }

	if (count > mMaxPoints)
	{
		count = mMaxPoints;
	}
    cl::BufferGL buffGL(mContext, CL_MEM_READ_WRITE, mPointBuff.buffer());

    cl::Event event;
    std::vector<cl::Memory> gl_objs = { buffGL };

    mKernel.setArg(0, points.buffer());
    mKernel.setArg(1, (int)count);
    mKernel.setArg(2, buffGL);
    mKernel.setArg(3, (int)(width/2));
    mKernel.setArg(4, (int)(height/2));
    size_t gSize = count+(16-(count%16));
    queue.enqueueAcquireGLObjects(&gl_objs);
    queue.enqueueNDRangeKernel(mKernel, cl::NullRange, cl::NDRange(gSize), cl::NullRange, NULL, &event);
    event.wait();
    queue.enqueueReleaseGLObjects(&gl_objs);

    mPainter.draw(GL_LINES, 0, (GLsizei)(count*4), mPointBuff);
}
