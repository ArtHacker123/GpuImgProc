#include "OglView.h"

#define MAX_CORNER_COUNT 10000

OglView::OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
    :mRvalue(0.00125f),
     mCtxtCL(ctxt),
     mQueueCL(queue),
     mYuvImg(w, h),
     mCorners(mCtxtCL, CL_MEM_READ_WRITE, MAX_CORNER_COUNT),
     mCornerCount(mCtxtCL, CL_MEM_READ_WRITE, 1),
     mHarrisCorner(mCtxtCL),
     mCornerPainter(mCtxtCL, MAX_CORNER_COUNT)
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

    cl::ImageGL imgGL(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mYuvImg.yImage().texture());

    std::vector<cl::Event> events;
    mHarrisCorner.process(mQueueCL, imgGL, mCorners, mRvalue, mCornerCount, events);
    events.back().wait();

    cl_int count = 0;
    mQueueCL.enqueueReadBuffer(mCornerCount.buffer(), CL_TRUE, 0, sizeof(cl_int), &count);

    mYuvPainter.draw(mYuvImg);
    mCornerPainter.draw(mQueueCL, mCorners, count, mYuvImg.width(), mYuvImg.height());
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
