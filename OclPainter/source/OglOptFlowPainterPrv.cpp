#include "OglOptFlowPainterPrv.h"

#define OCL_PROGRAM_SOURCE(s) #s

using namespace Ogl;

const char OptFlowPainterPrv::sSource[] = OCL_PROGRAM_SOURCE(

typedef struct _OptFlowData
{
    int x;
    int y;
    float u;
    float v;
} OptFlowData;

inline void cross_coords(int i, int2 pos, int width, int height, global float2* coord, float2 uv)
{
    float r = length(uv);
    coord[i] = (float2)(((float)pos.x)/(float)width, ((float)pos.y)/(float)height);
    coord[i+1] = coord[i] + uv;
    coord[i+2] = coord[i+1];
    coord[i+3] = coord[i+1]-(float2)((4.0*(uv.x+uv.y)*cos(M_PI_F/4.0))/((float)width*r), (4.0*(uv.y-uv.x)*cos(M_PI_F/4.0))/((float)height*r));
    coord[i+4] = coord[i+1];
    coord[i+5] = coord[i+1]-(float2)((4.0*(uv.x-uv.y)*cos(M_PI_F/4.0))/((float)width*r), (4.0*(uv.y+uv.x)*cos(M_PI_F/4.0))/((float)height*r));
}

kernel void find_coords(global OptFlowData* input, global float2* coord, int max_size, int width, int height)
{
    int i = get_global_id(0);
    if (i < max_size)
    {
        OptFlowData fd = input[i];
        int2 pos = (int2)(fd.x-width, height- fd.y);
        float2 uv = (float2)(fd.u/(float)(width<<1), fd.v/(float)(height<<1));
        cross_coords((i*6), pos, width, height, coord, uv);
    }
}

);

OptFlowPainterPrv::OptFlowPainterPrv(const cl::Context& ctxt, size_t maxSize)
    :mMaxSize(maxSize),
     mContext(ctxt),
     mOptFlowBuff(GL_ARRAY_BUFFER, (12*mMaxSize*sizeof(GLfloat)), 0, GL_DYNAMIC_DRAW)
{
    try
    {
        cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
        mPgm = cl::Program(mContext, source);
        mPgm.build();
        mKernel = cl::Kernel(mPgm, "find_coords");
        mPainter.SetColor(1.0f, 0.0f, 0.0f, 1.0f);
    }

    catch (cl::Error error)
    {
        printf("\nFailed: %s", error.what());
        exit(0);
    }
}

OptFlowPainterPrv::~OptFlowPainterPrv()
{
}

void OptFlowPainterPrv::draw(const cl::CommandQueue& queue, Ocl::DataBuffer<Ocl::OptFlowData>& flowData, size_t count, size_t width, size_t height)
{
    if (count <= 0)
    {
        return;
    }

    if (count > mMaxSize)
    {
        count = mMaxSize;
    }
    cl::BufferGL buffGL(mContext, CL_MEM_READ_WRITE, mOptFlowBuff.buffer());

    cl::Event event;
    std::vector<cl::Memory> gl_objs = { buffGL };

    mKernel.setArg(0, flowData.buffer());
    mKernel.setArg(1, buffGL);
    mKernel.setArg(2, (int)count);
    mKernel.setArg(3, (int)(width/2));
    mKernel.setArg(4, (int)(height/2));
    size_t gSize = count+(16-(count%16));
    queue.enqueueAcquireGLObjects(&gl_objs);
    queue.enqueueNDRangeKernel(mKernel, cl::NullRange, cl::NDRange(gSize), cl::NullRange, NULL, &event);
    event.wait();
    queue.enqueueReleaseGLObjects(&gl_objs);

    mPainter.draw(GL_LINES, 0, (GLsizei)(count*6), mOptFlowBuff);
}
