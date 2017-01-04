#include "OglHoughLinePainterPrv.h"

#define OCL_PROGRAM_SOURCE(s) #s

using namespace Ogl;

const char HoughLinePainterPrv::sSource[] = OCL_PROGRAM_SOURCE(

typedef struct _HoughLineData
{
    int rho;
    int angle;
    int strength;
} HoughLineData;

kernel void find_coords(global HoughLineData* input, const int max_size, global float2* coord, const float w, const float h)
{
    int i = get_global_id(0);
    if (i >= max_size) return;
    int count = 0;
    int offset = i*2;
    HoughLineData hd = input[i];
    float theta = (((float)hd.angle)/180.0f)*M_PI_F;
    if (hd.angle != 0 && hd.angle != 180)
    {
        float y = (((float)hd.rho)-(w*cos(theta)/2.0))/sin(theta);
        y /= (h/2);
        if (y >= -1.0f && y <= 1.0f)
        {
            coord[offset+count++] = (float2)(1.0f, y);
        }

        y = (((float)hd.rho)+(w*cos(theta)/2.0))/sin(theta);
        y /= (h/2);
        if (y >= -1.0f && y <= 1.0f)
        {
            coord[offset+count++] = (float2)(-1.0f, y);
        }
    }

    if (count == 2) return;

    if (hd.angle != 90 && hd.angle != 270)
    {
        float x = (((float)hd.rho)-(h*sin(theta)/2.0))/cos(theta);
        x /= (w/2);
        if (x >= -1.0f && x <= 1.0f)
        {
            coord[offset+count++] = (float2)(x, 1.0f);
        }

        if (count == 2) return;

        x = (((float)hd.rho)+(h*sin(theta)/2.0))/cos(theta);
        x /= (w/2);
        if (x >= -1.0f && x <= 1.0f)
        {
            coord[offset+count++] = (float2)(x, -1.0f);
        }
    }
}

);

HoughLinePainterPrv::HoughLinePainterPrv(cl::Context& ctxt, cl::CommandQueue& queue, size_t maxSize)
    :mMaxSize(maxSize),
     mContext(ctxt),
     mQueue(queue),
     mHoughLineBuff(GL_ARRAY_BUFFER, (4*mMaxSize*sizeof(GLfloat)), 0, GL_DYNAMIC_DRAW)
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

HoughLinePainterPrv::~HoughLinePainterPrv()
{
}

void HoughLinePainterPrv::draw(Ocl::DataBuffer<Ocl::HoughData>& hData, size_t count, size_t width, size_t height)
{
    if (count <= 0)
    {
        return;
    }

    if (count > mMaxSize)
    {
        count = mMaxSize;
    }
    cl::BufferGL buffGL(mContext, CL_MEM_READ_WRITE, mHoughLineBuff.buffer());

    cl::Event event;
    std::vector<cl::Memory> gl_objs = { buffGL };

    mKernel.setArg(0, hData.buffer());
    mKernel.setArg(1, (int)count);
    mKernel.setArg(2, buffGL);
    mKernel.setArg(3, (float)width);
    mKernel.setArg(4, (float)height);
    size_t gSize = count+(16-(count%16));
    mQueue.enqueueAcquireGLObjects(&gl_objs);
    mQueue.enqueueNDRangeKernel(mKernel, cl::NullRange, cl::NDRange(gSize), cl::NullRange, NULL, &event);
    event.wait();
    mQueue.enqueueReleaseGLObjects(&gl_objs);

    mPainter.draw(GL_LINES, 0, (GLsizei)(count*2), mHoughLineBuff);
}
