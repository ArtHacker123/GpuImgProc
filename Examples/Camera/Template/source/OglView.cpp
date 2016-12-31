#include "OglView.h"
#include "OglImageFormat.h"

#define OCL_PROGRAM_SOURCE(s) #s

const char sSource[] = OCL_PROGRAM_SOURCE(

kernel void test_kernel(read_only image2d_t inpImg, write_only image2d_t outImg)
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float3 idata = read_imagef(inpImg, coord).xyz;
    float gray = (idata.x+idata.y+idata.z)/3.0f;
    write_imagef(outImg, coord, gray);
}

);

OglView::OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
    :mCtxtCL(ctxt),
     mQueueCL(queue),
     mBgrImg(w, h, GL_RGB, GL_UNSIGNED_BYTE),
     mGrayImg(w, h, GL_R32F, GL_FLOAT)
{
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
    mBgrImg.load(pData);

    cl::ImageGL inpImgGL(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mBgrImg.texture());
    cl::ImageGL outImgGL(mCtxtCL, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, mGrayImg.texture());

    std::vector<cl::Memory> gl_objs = { inpImgGL, outImgGL };

    cl::Event event;
    mKernel.setArg(0, inpImgGL);
    mKernel.setArg(1, outImgGL);
    mQueueCL.enqueueAcquireGLObjects(&gl_objs);
    mQueueCL.enqueueNDRangeKernel(mKernel, cl::NullRange, cl::NDRange(mBgrImg.width(), mBgrImg.height()), cl::NDRange(8, 8), NULL, &event);
    event.wait();
    mQueueCL.enqueueReleaseGLObjects(&gl_objs);

    mRgbaPainter.draw(mBgrImg);
    Ogl::IGeometry::Rect vp = { mBgrImg.width()/2, 0, mBgrImg.width()/2, mBgrImg.height()/2 };
    mGrayPainter.draw(vp, mGrayImg);
}

void OglView::resize(GLsizei w, GLsizei h)
{
    glViewport(0, 0, w, h);
}
