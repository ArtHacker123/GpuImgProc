#include "OglView.h"

#define PYRAMID_LEVELS 4

OglView::OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
    :mCtxtCL(ctxt),
     mQueueCL(queue),
     mRvalue(0.000000250f),
     mMinFlowSize(6.0f),
     m_pPrevImg(0),
     m_pCurrImg(0),
     mImg1(w, h, 0, true),
     mImg2(w, h, 0, true),
     mOptFlow(mCtxtCL, PYRAMID_LEVELS),
     mFlowData(mCtxtCL, CL_MEM_READ_WRITE, (w*h)/4),
     mFlowPainter(mCtxtCL, (w*h)/4)
{
    m_pPrevImg = &mImg1;
    m_pCurrImg = &mImg2;
}

OglView::~OglView()
{
}

void OglView::swap()
{
    Ogl::Yuv420Image* pTemp = m_pPrevImg;
    m_pPrevImg = m_pCurrImg;
    m_pCurrImg = pTemp;
}

void OglView::draw(uint8_t* pData)
{
    m_pCurrImg->load(pData);
    size_t outCount = 0;
    mOptFlow.process(mQueueCL, mFlowData, outCount, m_pCurrImg->yImage(), m_pPrevImg->yImage(), mRvalue, mMinFlowSize);

    mYuvPainter.draw(*m_pCurrImg);
    mFlowPainter.draw(mQueueCL, mFlowData, outCount, m_pCurrImg->width(), m_pCurrImg->height());

    swap();
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

void OglView::minFlowSizeUp()
{
    mMinFlowSize += 0.5f;
}

void OglView::minFlowSizeDown()
{
    if (mMinFlowSize > 1.5f)
    {
        mMinFlowSize -= 0.5f;
    }
}

