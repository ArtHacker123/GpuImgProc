#include <CL/cl2.hpp>

#include "OglView.h"
#include "OglImageFormat.h"
#include "OclUtils.h"

#include <sstream>

#define MAX_CORNER_COUNT 1000

OglView::OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
    :mCtxtCL(ctxt),
     mQueueCL(queue),
     mBgrImg(w, h, GL_RGB, GL_UNSIGNED_BYTE),
     mGrayImg(w, h, GL_R32F, GL_FLOAT),
     mIntImg1(mCtxtCL, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), w, h),
     mIntImg2(mCtxtCL, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), w, h),
     mPointBuff(GL_ARRAY_BUFFER, (8*MAX_CORNER_COUNT*sizeof(GLfloat)), 0, GL_DYNAMIC_DRAW),
     m_pIntBuffData(0),
     m_pCornerData(0),
     m_pCornerParam(0)
{
    init();

    m_pCornerParam = (CornerParams *)clSVMAlloc(mCtxtCL(), CL_MEM_READ_WRITE, sizeof(CornerParams), sizeof(cl_int4));
    m_pCornerParam->rvalue = 0.00125f;
    m_pCornerParam->cornerCount = 0;
    m_pCornerParam->maxCornerCount = MAX_CORNER_COUNT;

    size_t tempBuffSize = ((w/32)*(h/8))+256;
    m_pIntBuffData = (cl_int*)clSVMAlloc(mCtxtCL(), CL_MEM_READ_WRITE, tempBuffSize*sizeof(cl_int), sizeof(cl_int4));
    m_pCornerData = (cl_int2*)clSVMAlloc(mCtxtCL(), CL_MEM_READ_WRITE, m_pCornerParam->maxCornerCount*sizeof(cl_int2), sizeof(cl_int4));

    mPainter.SetColor(1.0f, 0.0f, 0.0f, 1.0f);
}

OglView::~OglView()
{
    clSVMFree(mCtxtCL(), m_pIntBuffData);
    clSVMFree(mCtxtCL(), m_pCornerData);
    clSVMFree(mCtxtCL(), m_pCornerParam);
    m_pIntBuffData = 0;
    m_pCornerData = 0;
    m_pCornerParam = 0;
}

void OglView::init()
{
    try
    {
        std::ostringstream options;
        options << "-cl-std=CL2.0 -DBLK_SIZE_X=" << 32 << " -DBLK_SIZE_Y=" << 8;

        mProgram = cl::Program(mCtxtCL, std::string(sSource));
        mProgram.build(options.str().c_str());

        mKernel = cl::Kernel(mProgram, "harrisCorner");
        mCrossKernel = cl::Kernel(mProgram, "extractCoords");
    }

    catch (cl::Error err)
    {
        printf("\nFailed: %s", err.what());
        exit(0);
    }
}

void OglView::detectCorners(const cl::ImageGL& inpImg)
{
    cl::Event event;
    std::vector<cl::Memory> gl_objs = { inpImg };

    mKernel.setArg(0, inpImg);
    mKernel.setArg(1, mIntImg1);
    mKernel.setArg(2, mIntImg2);
    mKernel.setArg(3, m_pCornerData);
    mKernel.setArg(4, m_pIntBuffData);
    mKernel.setArg(5, m_pCornerParam);
    mQueueCL.enqueueAcquireGLObjects(&gl_objs);
    mQueueCL.enqueueNDRangeKernel(mKernel, cl::NullRange, cl::NDRange(1), cl::NDRange(1), NULL, &event);
    event.wait();
    mQueueCL.enqueueReleaseGLObjects(&gl_objs);
    size_t time = Ocl::kernelExecTime(mQueueCL, event);
}

void OglView::drawCorners()
{
    cl::BufferGL buffGL(mCtxtCL, CL_MEM_READ_WRITE, mPointBuff.buffer());

    cl::Event event;
    std::vector<cl::Memory> gl_objs = { buffGL };

    clSetKernelArgSVMPointer(mCrossKernel(), 0, m_pCornerData);
    mCrossKernel.setArg(1, (int)m_pCornerParam->cornerCount);
    mCrossKernel.setArg(2, buffGL);
    mCrossKernel.setArg(3, (int)(mBgrImg.width()/2));
    mCrossKernel.setArg(4, (int)(mBgrImg.height()/2));
    size_t gSize = m_pCornerParam->cornerCount+(16-(m_pCornerParam->cornerCount%16));
    mQueueCL.enqueueAcquireGLObjects(&gl_objs);
    mQueueCL.enqueueNDRangeKernel(mCrossKernel, cl::NullRange, cl::NDRange(gSize), cl::NullRange, NULL, &event);
    event.wait();
    mQueueCL.enqueueReleaseGLObjects(&gl_objs);

    mPainter.draw(GL_LINES, 0, (GLsizei)(m_pCornerParam->cornerCount*4), mPointBuff);
}

void OglView::draw(uint8_t* pData)
{
    mBgrImg.load(pData);
    Ogl::ImageFormat::convert(mGrayImg, mBgrImg);

    cl::ImageGL inpImgGL(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mGrayImg.texture());
    detectCorners(inpImgGL);

    mRgbaPainter.draw(mBgrImg);

    if (m_pCornerParam->cornerCount > 0)
    {
        drawCorners();
    }
}

void OglView::resize(GLsizei w, GLsizei h)
{
    glViewport(0, 0, w, h);
}

void OglView::thresholdUp()
{
    m_pCornerParam->rvalue *= 2.0;
}

void OglView::thresholdDown()
{
    m_pCornerParam->rvalue /= 2.0;
}
