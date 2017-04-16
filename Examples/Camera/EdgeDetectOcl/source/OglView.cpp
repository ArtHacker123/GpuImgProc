#include "OglView.h"
#include "OglImageFormat.h"
#include "OclUtils.h"

#define THRESHOLD_CHANGE 0.001f

OglView::OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
    :mMinThresh((float)(20.0/256.0)),
     mMaxThresh((float)(70.0/256.0)),
     mCtxtCL(ctxt),
     mQueueCL(queue),
     mCanny(mCtxtCL),
     mBgrImg(w, h, GL_RGB, GL_UNSIGNED_BYTE),
     mGrayImg(w, h, GL_R32F, GL_FLOAT),
     mEdgeImg(w, h, GL_R32F, GL_FLOAT)
{
}

OglView::~OglView()
{
}

void OglView::draw(uint8_t* pData)
{
    mBgrImg.load(pData);
    Ogl::ImageFormat::convert(mGrayImg, mBgrImg);

    std::vector<cl::Event> events;
    events.reserve(1024);

    cl::ImageGL inpImgGL(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mGrayImg.texture());
    cl::ImageGL outImgGL(mCtxtCL, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, mEdgeImg.texture());
    mCanny.process(mQueueCL, inpImgGL, outImgGL, mMinThresh, mMaxThresh, events);
    events.back().wait();
    size_t time = Ocl::kernelExecTime(mQueueCL, events.data(), events.size());

    mGrayPainter.draw(mEdgeImg);
    //Ogl::IGeometry::Rect vp = { mBgrImg.width()/2, 0, mBgrImg.width()/2, mBgrImg.height()/2 };
    //mRgbaPainter.draw(vp, mBgrImg);
}

void OglView::resize(GLsizei w, GLsizei h)
{
    glViewport(0, 0, w, h);
}

void OglView::minThresholdUp()
{
    float thresh = mMinThresh+THRESHOLD_CHANGE;
    if (thresh < mMaxThresh)
    {
        mMinThresh = thresh;
    }
}

void OglView::minThresholdDown()
{
    float thresh = mMinThresh-THRESHOLD_CHANGE;
    if (thresh > 0.0)
    {
        mMinThresh = thresh;
    }
}

void OglView::maxThresholdUp()
{
    float thresh = mMaxThresh+THRESHOLD_CHANGE;
    if (thresh < 1.0)
    {
        mMaxThresh = thresh;
    }
}

void OglView::maxThresholdDown()
{
    float thresh = mMaxThresh-THRESHOLD_CHANGE;
    if (thresh > mMinThresh)
    {
        mMaxThresh = thresh;
    }
}
