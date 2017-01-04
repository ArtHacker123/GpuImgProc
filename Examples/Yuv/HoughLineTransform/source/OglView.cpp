#include "OglView.h"

OglView::OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
    :mMinThresh((float)(20.0/256.0)),
     mMaxThresh((float)(70.0/256.0)),
     mCtxtCL(ctxt),
     mQueueCL(queue),
     mCanny(mCtxtCL, mQueueCL),
     mCompact(mCtxtCL, mQueueCL),
     mHoughLines(mCtxtCL, mQueueCL),
     mYuvImg(w, h),
     mEdgeImg(w, h, GL_R32F, GL_FLOAT),
     mHoughData(mCtxtCL, mQueueCL, w, h),
     mEdgeCoords(mCtxtCL, CL_MEM_READ_WRITE, (w*h)>>2)
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
    std::vector<cl::Memory> gl_objs = { outImgGL };
    mQueueCL.enqueueAcquireGLObjects(&gl_objs);
    size_t edgeCount = 0;
    time += mCompact.process(outImgGL, mEdgeCoords, 1.0f, edgeCount);
    mQueueCL.enqueueReleaseGLObjects(&gl_objs);
    time += mHoughLines.process(mEdgeCoords, edgeCount, mEdgeImg.width(), mEdgeImg.height(), mHoughData);
    mYuvPainter.draw(mYuvImg);
    
    Ogl::IGeometry::Rect vp = { mYuvImg.width()>>1, 0, mYuvImg.width()>>1, mYuvImg.height()>>1 };
    mLumaPainter.draw(vp, mEdgeImg);
}

void OglView::resize(GLsizei w, GLsizei h)
{
    glViewport(0, 0, w, h);
}
