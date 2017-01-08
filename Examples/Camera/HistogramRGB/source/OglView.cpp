#include "OglView.h"
#include "OglImageFormat.h"

OglView::OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
    :mCtxtCL(ctxt),
     mQueueCL(queue),
     mBgrImg(w, h, GL_RGBA32F, GL_UNSIGNED_BYTE),
     mRgbBins(mCtxtCL, CL_MEM_READ_WRITE, 768),
     mHistogram(mCtxtCL),
     mHistPainter(mCtxtCL)
{
    mRedBuff = mRgbBins.createSubBuffer(CL_MEM_READ_ONLY, 0, 256);
    mGreenBuff = mRgbBins.createSubBuffer(CL_MEM_READ_ONLY, 256, 256);
    mBlueBuff = mRgbBins.createSubBuffer(CL_MEM_READ_ONLY, 512, 256);
}

OglView::~OglView()
{
}

void OglView::draw(uint8_t* pData)
{
    mBgrImg.load(pData);

    cl::ImageGL imgGL(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mBgrImg.texture());
    size_t time = mHistogram.compute(mQueueCL, imgGL, mRgbBins);

    mPainter.draw(mBgrImg);

    int maxValue = (mBgrImg.width()*mBgrImg.height()>>4);

    mHistPainter.setColor(1.0f, 0.0f, 0.0f);
    mHistPainter.draw(mQueueCL, mRedBuff, maxValue);

    mHistPainter.setColor(0.0f, 1.0f, 0.0f);
    mHistPainter.draw(mQueueCL, mGreenBuff, maxValue);

    mHistPainter.setColor(0.0f, 0.0f, 1.0f);
    mHistPainter.draw(mQueueCL, mBlueBuff, maxValue);
}

void OglView::resize(GLsizei w, GLsizei h)
{
    glViewport(0, 0, w, h);
}
