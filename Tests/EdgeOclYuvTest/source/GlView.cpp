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

    //Ogl::Image<GL_RG> image(640, 480, GL_RGBA32F, GL_UNSIGNED_BYTE);
    //cl::ImageGL test(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, image.texture());

    cl::ImageGL outImgGL(mCtxtCL, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, mEdgeImg.texture());
    cl::ImageGL inpImgGL(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mYuvImg.yImage().texture());

    mCanny.process(inpImgGL, outImgGL, minThresh, maxThresh);

	//mYuvPainter.draw(mYuvImg);
    mLumaPainter.draw(mEdgeImg);

    //Ogl::IGeometry::Rect viewPort = { mYuvImg.width() >> 1, 0, mYuvImg.width() >> 1, mYuvImg.height() >> 1 };
    //mLumaPainter.draw(viewPort, mEdgeImg);
}

void GlView::resize(GLsizei w, GLsizei h)
{
	glViewport(0, 0, w, h);
}
