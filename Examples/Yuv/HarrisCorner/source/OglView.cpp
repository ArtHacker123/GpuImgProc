#include "OglView.h"

#define MAX_CORNER_COUNT 10000

OglView::OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
    :mCtxtCL(ctxt),
     mQueueCL(queue),
     mYuvImg(w, h),
     mCorners(mCtxtCL, CL_MEM_READ_WRITE, MAX_CORNER_COUNT),
     mHarrisCorner(mCtxtCL, mQueueCL),
     mCornerPainter(mCtxtCL, mQueueCL, MAX_CORNER_COUNT)
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

OglView::~OglView()
{
}

void OglView::draw(uint8_t* pData)
{
    mYuvImg.load(pData);

    size_t count = 0;
    cl::ImageGL imgGL(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mYuvImg.yImage().texture());
    mHarrisCorner.process(imgGL, mCorners, 0.00125f, count);

    mYuvPainter.draw(mYuvImg);
    mCornerPainter.draw(mCorners, count, mYuvImg.width(), mYuvImg.height());
}

void OglView::resize(GLsizei w, GLsizei h)
{
    glViewport(0, 0, w, h);
}
