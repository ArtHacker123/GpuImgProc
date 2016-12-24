#include "GlView.h"

GlView::GlView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
    :mCtxtCL(ctxt),
     mQueueCL(queue),
     mCanny(ctxt, queue),
     mYuvImg(w, h),
     mEdgeImg(w, h, GL_R32F, GL_FLOAT),
     minThresh((float)(20.0/256.0)),
     maxThresh((float)(70.0/256.0))
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

GlView::~GlView()
{
}

void GlView::draw(uint8_t* pData)
{
    mYuvImg.load(pData);

    cl::ImageGL outImgGL(mCtxtCL, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, mEdgeImg.texture());
    cl::ImageGL inpImgGL(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mYuvImg.yImage().texture());

    mCanny.process(inpImgGL, outImgGL, minThresh, maxThresh);

    mYuvPainter.draw(mYuvImg);
    
    Ogl::IGeometry::Rect vp = { mYuvImg.width()>>1, 0, mYuvImg.width()>>1, mYuvImg.height()>>1 };
    mLumaPainter.draw(vp, mEdgeImg);
}

void GlView::resize(GLsizei w, GLsizei h)
{
    glViewport(0, 0, w, h);
}
