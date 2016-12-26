#include "OglView.h"
#include "OglFrameBuffer.h"
#include "OglImageFormat.h"

#define PYRAMID_LEVELS 4

OglView::OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue, uint8_t* pData)
    :mCtxtCL(ctxt),
     mQueueCL(queue),
     mLineCount(0),
     m_pPrevImg(0),
     m_pCurrImg(0),
     mImg1(w, h, GL_R32F, GL_FLOAT, 0, true),
     mImg2(w, h, GL_R32F, GL_FLOAT, 0, true),
     mBgrImg(w, h, GL_RGBA32F, GL_UNSIGNED_BYTE),
     mOptFlow(PYRAMID_LEVELS),
     mBuffer(GL_ARRAY_BUFFER, (w*h*6), 0, GL_DYNAMIC_DRAW),
     mCoordExtract(mCtxtCL, mQueueCL)
{
    m_pPrevImg = &mImg1;
    m_pCurrImg = &mImg2;
    draw(pData);
}

OglView::~OglView()
{
}

void OglView::drawOptFlow()
{
    cl::BufferGL buffGL(mCtxtCL, CL_MEM_READ_WRITE, mBuffer.buffer());
    cl::ImageGL imgGL(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mUVImg[0]->texture());
    mCoordExtract.do_work(imgGL, buffGL);
    GLsizei count = 6*(mUVImg[0]->width()/4)*(mUVImg[0]->height()/4);
    mPainter.draw(GL_LINES, 0, count, mBuffer);
}

void OglView::draw(uint8_t* pData)
{
    mBgrImg.load(pData);
    Ogl::ImageFormat::convert(*m_pCurrImg, mBgrImg);

    mOptFlow.process(mUVImg, *m_pCurrImg, *m_pPrevImg);

    mBgrPainter.draw(mBgrImg);

    drawOptFlow();

    Ogl::Image<GL_RED>* pTemp = m_pPrevImg;
    m_pPrevImg = m_pCurrImg;
    m_pCurrImg = pTemp;
}

void OglView::resize(GLsizei w, GLsizei h)
{
    glViewport(0, 0, w, h);
}
