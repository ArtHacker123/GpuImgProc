#include "OglView.h"
#include "OglImageFormat.h"
#include "OclUtils.h"

#define THRESHOLD_CHANGE 0.001f

OglView::OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
    :mSize((size_t)(w/4)),
     mMinThresh((float)(20.0/256.0)),
     mMaxThresh((float)(70.0/256.0)),
     mCtxtCL(ctxt),
     mQueueCL(queue),
     mBgrImg(w, h, GL_RGB, GL_UNSIGNED_BYTE),
     mGrayImg(w, h, GL_R32F, GL_FLOAT),
     mEdgeImg(w, h, GL_R32F, GL_FLOAT),
     mCanny(mCtxtCL),
     mHoughLines(mCtxtCL),
     mLineCount(mCtxtCL, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 1),
     mHoughData(mCtxtCL, CL_MEM_READ_WRITE, 1000),
     mHoughLinePainter(mCtxtCL, 1000)
{
}

OglView::~OglView()
{
}

void OglView::draw(uint8_t* pData)
{
    mBgrImg.load(pData);
    Ogl::ImageFormat::convert(mGrayImg, mBgrImg);

    cl::ImageGL inpImgGL(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mGrayImg.texture());
    cl::ImageGL outImgGL(mCtxtCL, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, mEdgeImg.texture());

    std::vector<cl::Event> events;
    events.reserve(1024);
    mCanny.process(mQueueCL, inpImgGL, outImgGL, mMinThresh, mMaxThresh, events);
    std::vector<cl::Event> mWaitEvent = { events.back() };
    mHoughLines.process(mQueueCL, outImgGL, mSize, mHoughData, mLineCount, events, &mWaitEvent);
    events.back().wait();
    size_t time = Ocl::kernelExecTime(mQueueCL, events.data(), events.size());

    mRgbaPainter.draw(mBgrImg);
    //mGrayPainter.draw(mEdgeImg);
    cl_int lines = 0;
    mQueueCL.enqueueReadBuffer(mLineCount.buffer(), CL_TRUE, 0, sizeof(cl_int), &lines);
    mHoughLinePainter.draw(mQueueCL, mHoughData, lines, mEdgeImg.width(), mEdgeImg.height());

    //Ogl::IGeometry::Rect vp = { mBgrImg.width()/2, 0, mBgrImg.width()/2, mBgrImg.height()/2 };
    //mGrayPainter.draw(vp, mEdgeImg);
}

void OglView::resize(GLsizei w, GLsizei h)
{
    glViewport(0, 0, w, h);
}

void OglView::minThresholdUp()
{
    float thresh = mMinThresh + THRESHOLD_CHANGE;
    if (thresh < mMaxThresh)
    {
        mMinThresh = thresh;
    }
}

void OglView::minThresholdDown()
{
    float thresh = mMinThresh - THRESHOLD_CHANGE;
    if (thresh > 0.0)
    {
        mMinThresh = thresh;
    }
}

void OglView::maxThresholdUp()
{
    float thresh = mMaxThresh + THRESHOLD_CHANGE;
    if (thresh < 1.0)
    {
        mMaxThresh = thresh;
    }
}

void OglView::maxThresholdDown()
{
    float thresh = mMaxThresh - THRESHOLD_CHANGE;
    if (thresh > mMinThresh)
    {
        mMaxThresh = thresh;
    }
}

void OglView::minLineSizeUp()
{
    mSize += 10;
}

void OglView::minLineSizeDown()
{
    if (mSize > 20)
    {
        mSize -= 10;
    }
}
