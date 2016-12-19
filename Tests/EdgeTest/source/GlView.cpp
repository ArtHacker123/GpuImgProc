#include "GlView.h"
#include "OglCanny.h"

GlView::GlView(GLsizei w, GLsizei h)
	:minThresh((GLfloat)(20.0/256.0)),
     maxThresh((GLfloat)(70.0/256.0)),
     mYuvImg(w, h),
	 mEdgeImg(w, h, GL_R32F, GL_FLOAT)
{
    Ogl::Canny::init();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

GlView::~GlView()
{
}

void GlView::draw(uint8_t* pData)
{
	mYuvImg.load(pData);
	Ogl::Canny::process(mEdgeImg, mYuvImg.yImage(), minThresh, maxThresh);

	glClear(GL_COLOR_BUFFER_BIT);
	mYuvPainter.draw(mYuvImg);

    Ogl::IGeometry::Rect viewPort = { 0, mYuvImg.width()>>1, mYuvImg.width()>>1, mYuvImg.height()>>1 };
	mLumaPainter.draw(viewPort, mEdgeImg);
}

void GlView::resize(GLsizei w, GLsizei h)
{
	glViewport(0, 0, w, h);
}

void GlView::keyDown(KEYS key)
{
	// TODO: Add your message handler code here and/or call default
	switch (key)
	{
		case UP:
			if (maxThresh < 1.0)
			{
				maxThresh += 0.01;
			}
			break;
		case DOWN:
			if (minThresh < maxThresh)
			{
				maxThresh -= 0.01;
			}
			break;
		case LEFT:
			if (minThresh > 0.0)
			{
				minThresh -= 0.01;
			}
			break;
		case RIGHT:
			if (minThresh < maxThresh)
			{
				minThresh += 0.01;
			}
			break;
	}
}
