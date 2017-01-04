#include "OglView.h"

OglView::OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
    :mMinThresh((float)(20.0/256.0)),
     mMaxThresh((float)(70.0/256.0)),
     mCtxtCL(ctxt),
     mQueueCL(queue),
     mCanny(mCtxtCL, mQueueCL),
     mHoughLines(mCtxtCL, mQueueCL),
     mYuvImg(w, h),
     mEdgeImg(w, h, GL_R32F, GL_FLOAT),
     mHoughData(mCtxtCL, CL_MEM_READ_WRITE, 200),
     mHoughLinePainter(mCtxtCL, mQueueCL, 200)
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    /*Ocl::HoughData* pHoughData = mHoughData.map(mQueueCL, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, 100);

    pHoughData[0].angle = 0;
    pHoughData[0].rho = 100;
    pHoughData[0].strength = 200;

    pHoughData[1].angle = 90;
    pHoughData[1].rho = 70;
    pHoughData[1].strength = 200;

    pHoughData[2].angle = 180;
    pHoughData[2].rho = 100;
    pHoughData[2].strength = 200;

    pHoughData[3].angle = 270;
    pHoughData[3].rho = 70;
    pHoughData[3].strength = 200;

    mHoughData.unmap(mQueueCL, pHoughData);*/
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
    time += mHoughLines.process(outImgGL, 50, mHoughData, houghCount);

    mYuvPainter.draw(mYuvImg);  
    mHoughLinePainter.draw(mHoughData, houghCount, mEdgeImg.width(), mEdgeImg.height());
    //Ogl::IGeometry::Rect vp = { mYuvImg.width()>>1, 0, mYuvImg.width()>>1, mYuvImg.height()>>1 };
    //mLumaPainter.draw(vp, mEdgeImg);
}

void OglView::resize(GLsizei w, GLsizei h)
{
    glViewport(0, 0, w, h);
}
