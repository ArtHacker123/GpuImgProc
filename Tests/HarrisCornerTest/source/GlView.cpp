#include "GlView.h"
#include "OglImageFormat.h"

GlView::GlView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue)
	:mCtxtCL(ctxt),
	 mQueueCL(queue),
	 mBgrImg(w, h, GL_RGB, GL_UNSIGNED_BYTE),
	 mGrayImg(w, h, GL_R32F, GL_FLOAT),
	 mCornerPaint(mCtxtCL, mQueueCL),
	 mHarrisCorner(mCtxtCL, mQueueCL),
	 mCorners(mCtxtCL, CL_MEM_READ_WRITE, 1000)
{
}

GlView::~GlView()
{
}

void GlView::draw(uint8_t* pData)
{
	mBgrImg.load(pData);
    Ogl::ImageFormat::convert(mGrayImg, mBgrImg);

    size_t count = 0;
    cl::ImageGL imgGL(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mGrayImg.texture());
	mHarrisCorner.process(imgGL, mCorners, 0.00125f, count);

	mBgrPainter.draw(mBgrImg);
	mCornerPaint.draw(mCorners, mBgrImg.width(), mBgrImg.height(), count);
}

void GlView::resize(GLsizei w, GLsizei h)
{
	glViewport(0, 0, w, h);
}
