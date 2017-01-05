#include "OglView.h"

#define THRESHOLD_CHANGE 0.001f

OglView::OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
    :mSize((size_t)(w/4)),
     mMinThresh((float)(20.0/256.0)),
     mMaxThresh((float)(60.0/256.0)),
     mCtxtCL(ctxt),
     mQueueCL(queue),
     mCanny(mCtxtCL, mQueueCL),
     mHoughLines(mCtxtCL, mQueueCL),
     mYuvImg(w, h),
     mEdgeImg(w, h, GL_R32F, GL_FLOAT),
     mHoughData(mCtxtCL, CL_MEM_READ_WRITE, 2000),
     mHoughLinePainter(mCtxtCL, mQueueCL, 2000)
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

    cl::ImageGL outImgGL(mCtxtCL, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, mEdgeImg.texture());
    cl::ImageGL inpImgGL(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mYuvImg.yImage().texture());

    size_t time = mCanny.process(inpImgGL, outImgGL, mMinThresh, mMaxThresh);

    size_t houghCount = 0;
    time += mHoughLines.process(outImgGL, mSize, mHoughData, houghCount);

    mYuvPainter.draw(mYuvImg);
    mHoughLinePainter.draw(mHoughData, houghCount, mEdgeImg.width(), mEdgeImg.height());
    //Ogl::IGeometry::Rect vp = { mYuvImg.width()>>1, 0, mYuvImg.width()>>1, mYuvImg.height()>>1 };
    //mLumaPainter.draw(vp, mEdgeImg);
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
