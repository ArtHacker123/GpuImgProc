#include "OglView.h"
#include "OglImageFormat.h"

#define MAX_CORNER_COUNT 10000

OglView::OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
    :mRvalue(0.00125f),
     mCtxtCL(ctxt),
     mQueueCL(queue),
     mBgrImg(w, h, GL_RGB, GL_UNSIGNED_BYTE),
     mGrayImg(w, h, GL_R32F, GL_FLOAT),
     mCorners(mCtxtCL, CL_MEM_READ_WRITE, MAX_CORNER_COUNT),
     mHarrisCorner(mCtxtCL),
     mCornerPainter(mCtxtCL, MAX_CORNER_COUNT)
{
}

OglView::~OglView()
{
}

void OglView::draw(uint8_t* pData)
{
    mBgrImg.load(pData);
    Ogl::ImageFormat::convert(mGrayImg, mBgrImg);

    size_t count = 0;
    cl::ImageGL imgGL(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mGrayImg.texture());
    mHarrisCorner.process(mQueueCL, imgGL, mCorners, mRvalue, count);

    mBgrPainter.draw(mBgrImg);
    mCornerPainter.draw(mQueueCL, mCorners, count, mBgrImg.width(), mBgrImg.height());
}

void OglView::resize(GLsizei w, GLsizei h)
{
    glViewport(0, 0, w, h);
}

void OglView::thresholdUp()
{
    mRvalue *= 2.0;
}

void OglView::thresholdDown()
{
    mRvalue /= 2.0;
}
