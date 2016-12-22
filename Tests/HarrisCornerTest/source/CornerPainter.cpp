#include "CornerPainter.h"

#define OCL_PROGRAM_SOURCE(s) #s

const char CornerPainter::sSource[] = OCL_PROGRAM_SOURCE(
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

CornerPainter::CornerPainter(cl::Context& ctxt, cl::CommandQueue& queue, size_t maxCorners)
    :mMaxCorners(maxCorners),
	 mContext(ctxt),
     mQueue(queue),
	 mCornerBuff(GL_ARRAY_BUFFER, (8*mMaxCorners*sizeof(GLfloat)), 0, GL_DYNAMIC_DRAW)
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

CornerPainter::~CornerPainter()
{
}

void CornerPainter::draw(Ocl::DataBuffer<Ocl::Pos>& corners, size_t width, size_t height, size_t count)
{
    if (count <= 0)
    {
        return;
    }

	if (count > mMaxCorners)
	{
		count = mMaxCorners;
	}
    cl::BufferGL buffGL(mContext, CL_MEM_READ_WRITE, mCornerBuff.buffer());

    cl::Event event;
    std::vector<cl::Memory> gl_objs = { buffGL };

    mKernel.setArg(0, corners.buffer());
    mKernel.setArg(1, (int)count);
    mKernel.setArg(2, buffGL);
    mKernel.setArg(3, (int)(width/2));
    mKernel.setArg(4, (int)(height/2));
    size_t gSize = count+(16-(count%16));
    mQueue.enqueueAcquireGLObjects(&gl_objs);
    mQueue.enqueueNDRangeKernel(mKernel, cl::NullRange, cl::NDRange(gSize), cl::NullRange, NULL, &event);
    event.wait();
    mQueue.enqueueReleaseGLObjects(&gl_objs);

    mPainter.draw(GL_LINES, 0, (GLsizei)(count*4), mCornerBuff);
}
