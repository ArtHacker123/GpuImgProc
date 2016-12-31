#include "OglView.h"

#define OCL_PROGRAM_SOURCE(s) #s

const char sSource[] = OCL_PROGRAM_SOURCE(

    kernel void test_kernel(read_only image2d_t inpImg, write_only image2d_t outImg)
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float idata = 1.0f-read_imagef(inpImg, coord).x;
    write_imagef(outImg, coord, idata);
}

);

OglView::OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
    :mCtxtCL(ctxt),
     mQueueCL(queue),
     mYuvImg(w, h),
     mGrayImg(w, h, GL_R32F, GL_FLOAT)
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
    mProgram = cl::Program(mCtxtCL, source);
    mProgram.build();
    mKernel = cl::Kernel(mProgram, "test_kernel");
}

OglView::~OglView()
{
}

void OglView::draw(uint8_t* pData)
{
    mYuvImg.load(pData);

    cl::ImageGL outImgGL(mCtxtCL, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, mGrayImg.texture());
    cl::ImageGL inpImgGL(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mYuvImg.yImage().texture());

    std::vector<cl::Memory> gl_objs = { inpImgGL, outImgGL };

    cl::Event event;
    mKernel.setArg(0, inpImgGL);
    mKernel.setArg(1, outImgGL);
    mQueueCL.enqueueAcquireGLObjects(&gl_objs);
    mQueueCL.enqueueNDRangeKernel(mKernel, cl::NullRange, cl::NDRange(mGrayImg.width(), mGrayImg.height()), cl::NDRange(8, 8), NULL, &event);
    event.wait();
    mQueueCL.enqueueReleaseGLObjects(&gl_objs);

    mYuvPainter.draw(mYuvImg);
    
    Ogl::IGeometry::Rect vp = { mYuvImg.width()>>1, 0, mYuvImg.width()>>1, mYuvImg.height()>>1 };
    mLumaPainter.draw(vp, mGrayImg);
}

void OglView::resize(GLsizei w, GLsizei h)
{
    glViewport(0, 0, w, h);
}
