#include "GlView.h"
#include "OglFrameBuffer.h"
#include "OglImageFormat.h"

#define _USE_MATH_DEFINES
#include <math.h>

GlView::GlView(GLsizei w, GLsizei h, cl::Context& clContext, cl::CommandQueue& clQueue)
	:mClContext(clContext),
	 mClQueue(clQueue),
     mBgrImg(w, h, GL_RGB, GL_UNSIGNED_BYTE),
     mRgbaImg(w, h, GL_RGBA, GL_UNSIGNED_BYTE),
     mRgbBins(mClContext, CL_MEM_READ_WRITE, 3*256),
	 mHistogram(mClContext, mClQueue),
     mHistPainter(mClContext, mClQueue)
{
    cl_buffer_region region = { 0, 256*sizeof(int) };
    mRedBuff = mRgbBins.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region);
    region.origin = 256*sizeof(int);
    mGreenBuff = mRgbBins.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region);
    region.origin = 512*sizeof(int);
    mBlueBuff = mRgbBins.createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region);
}

GlView::~GlView()
{
}

void GlView::draw(uint8_t* pData)
{
	mBgrImg.load(pData);
    Ogl::ImageFormat::convert(mRgbaImg, mBgrImg);

    cl::ImageGL imgGL(mClContext, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mRgbaImg.texture());
    size_t time = mHistogram.compute(imgGL, mRgbBins);

    //glClear(GL_COLOR_BUFFER_BIT);
    mPainter.draw(mBgrImg);

    int maxValue = (mBgrImg.width()*mBgrImg.height()>>4);

    mHistPainter.setColor(1.0f, 0.0f, 0.0f);
    mHistPainter.draw(mRedBuff, maxValue);

    mHistPainter.setColor(0.0f, 1.0f, 0.0f);
    mHistPainter.draw(mGreenBuff, maxValue);

    mHistPainter.setColor(0.0f, 0.0f, 1.0f);
    mHistPainter.draw(mBlueBuff, maxValue);
}

void GlView::resize(GLsizei w, GLsizei h)
{
	glViewport(0, 0, w, h);
}
