#include "GlView.h"
#include "OglImageFormat.h"

GlView::GlView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
	:mCtxtCL(ctxt),
     mQueueCL(queue),
     mHistogram(ctxt, queue),
     mHistPainter(ctxt, queue),
     mRgbBins(mCtxtCL, CL_MEM_READ_WRITE, 256*3),
     mYuvImg(w, h),
	 mRgbaImg(w, h, GL_RGBA, GL_UNSIGNED_BYTE)
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

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
	mYuvImg.load(pData);
    Ogl::ImageFormat::convert(mRgbaImg, mYuvImg);

    cl::ImageGL imgGL(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mRgbaImg.texture());
    size_t time = mHistogram.compute(imgGL, mRgbBins);

	mRgbaPainter.draw(mRgbaImg);

    int maxValue = (mRgbaImg.width()*mRgbaImg.height()>>4);

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
