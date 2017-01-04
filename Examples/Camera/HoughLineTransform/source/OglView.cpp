#include "OglView.h"
#include "OglImageFormat.h"

OglView::OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
    :mMinThresh((float)(20.0/256.0)),
     mMaxThresh((float)(70.0/256.0)),
     mCtxtCL(ctxt),
     mQueueCL(queue),
     mBgrImg(w, h, GL_RGB, GL_UNSIGNED_BYTE),
     mGrayImg(w, h, GL_R32F, GL_FLOAT),
     mEdgeImg(w, h, GL_R32F, GL_FLOAT),
     mCanny(mCtxtCL, mQueueCL),
     mHoughLines(mCtxtCL, mQueueCL),
     mHoughData(mCtxtCL, CL_MEM_READ_WRITE, 1000),
     mHoughLinePainter(mCtxtCL, mQueueCL, 1000)
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

    size_t time = mCanny.process(inpImgGL, outImgGL, mMinThresh, mMaxThresh);    
    
    size_t lines = 0;
    time += mHoughLines.process(outImgGL, 200, mHoughData, lines);

    mRgbaPainter.draw(mBgrImg);
    mHoughLinePainter.draw(mHoughData, lines, mEdgeImg.width(), mEdgeImg.height());

    //Ogl::IGeometry::Rect vp = { mBgrImg.width()/2, 0, mBgrImg.width()/2, mBgrImg.height()/2 };
    //mGrayPainter.draw(vp, mEdgeImg);
}

void OglView::resize(GLsizei w, GLsizei h)
{
    glViewport(0, 0, w, h);
}
