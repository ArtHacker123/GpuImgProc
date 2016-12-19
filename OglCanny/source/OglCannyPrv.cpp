#include "OglCannyPrv.h"

#include "OglImage.h"
#include "OglShape.h"

#include "OglFrameBuffer.h"

#include "OglEdgeShader.h"
#include "OglNmesShader.h"
#include "OglGaussShader.h"
#include "OglBinaryShader.h"

using namespace Ogl;

const std::vector<GLushort> CannyPrv::index = { 0, 1, 2, 2, 3, 0 };

const std::vector<Ogl::GeometryCoord> CannyPrv::coords =
	{
		{ { -1.0f, +1.0f, +0.0f }, { +0.0f, +1.0f } },
		{ { +1.0f, +1.0f, +0.0f }, { +1.0f, +1.0f } },
		{ { +1.0f, -1.0f, +0.0f }, { +1.0f, +0.0f } },
		{ { -1.0f, -1.0f, +0.0f }, { +0.0f, +0.0f } }
	};

const GLenum CannyPrv::mBuffs[] =
	{
		GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2,
		GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5
	};

CannyPrv::CannyPrv()
{
	mFb.reset(new Ogl::FrameBuffer);

	mEdgeShader.reset(new Ogl::EdgeShader);
	mNmesShader.reset(new Ogl::NmesShader);
	mGaussShader.reset(new Ogl::GaussShader);
	mBinaryShader.reset(new Ogl::BinaryShader);

	GLint vcIndex = mGaussShader->GetAttribLocation("inVc");
	GLint tcIndex = mGaussShader->GetAttribLocation("inTc");
	mRect.reset(new Ogl::Shape(vcIndex, tcIndex, index, coords));
}

CannyPrv::~CannyPrv()
{
}

void CannyPrv::createImages(GLsizei w, GLsizei h)
{
	if (mEdgeImg.get() == 0 || w != mEdgeImg->width() || h != mEdgeImg->height())
	{
		mEdgeImg.reset(new Ogl::Image<GL_RED>(w, h, GL_R32F, GL_FLOAT));
		mAngleImg.reset(new Ogl::Image<GL_RED>(w, h, GL_R32F, GL_FLOAT));
		mSmoothImg.reset(new Ogl::Image<GL_RED>(w, h, GL_R32F, GL_FLOAT));
		mNmesEdgeImg.reset(new Ogl::Image<GL_RED>(w, h, GL_R32F, GL_FLOAT));
	}
}

void CannyPrv::doProcess(Ogl::Image<GL_RED>& destImg, const Ogl::IImage& srcImg, GLfloat minThresh, GLfloat maxThresh)
{
	GLint params[4];
	createImages(destImg.width(), destImg.height());
	glGetIntegerv(GL_VIEWPORT, params);
	glViewport(0, 0, destImg.width(), destImg.height());
	doGaussSmooth(*mSmoothImg, srcImg);
	doEdgeAngles(*mEdgeImg, *mAngleImg, *mSmoothImg);
	doEdgeSuppress(*mNmesEdgeImg, *mEdgeImg, *mAngleImg);
	doBinary(destImg, *mNmesEdgeImg, minThresh, maxThresh);
	glViewport(params[0], params[1], params[2], params[3]);
}

void CannyPrv::doGaussSmooth(Ogl::Image<GL_RED>& destImg, const Ogl::IImage& srcImg)
{
	mFb->bind(GL_DRAW_FRAMEBUFFER);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, destImg.texture(), 0);
	mRect->draw(*mGaussShader, srcImg);
	Ogl::FrameBuffer::release(GL_DRAW_FRAMEBUFFER);
}

void CannyPrv::doEdgeAngles(Ogl::Image<GL_RED>& edgeImg, Ogl::Image<GL_RED>& angleImg, const Ogl::IImage& srcImg)
{
	mFb->bind(GL_DRAW_FRAMEBUFFER);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, angleImg.texture(), 0);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, edgeImg.texture(), 0);
	mRect->draw(*mEdgeShader, srcImg);
	glDrawBuffers(2, mBuffs);
	Ogl::FrameBuffer::release(GL_DRAW_FRAMEBUFFER);
}

void CannyPrv::doEdgeSuppress(Ogl::Image<GL_RED>& destImg, const Ogl::Image<GL_RED>& edgeImg, const Ogl::Image<GL_RED>& angleImg)
{
	edgeImg.bind(GL_TEXTURE1);
	mFb->bind(GL_DRAW_FRAMEBUFFER);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, destImg.texture(), 0);
	mRect->draw(*mNmesShader, angleImg);
	Ogl::FrameBuffer::release(GL_DRAW_FRAMEBUFFER);
}

void CannyPrv::doBinary(Ogl::Image<GL_RED>& destImg, const Ogl::IImage& srcImg, GLfloat minThresh, GLfloat maxThresh)
{
    mBinaryShader->setMinThreshold(minThresh);
    mBinaryShader->setMaxThreshold(maxThresh);
	mFb->bind(GL_DRAW_FRAMEBUFFER);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, destImg.texture(), 0);
	mRect->draw(*mBinaryShader, srcImg);
	Ogl::FrameBuffer::release(GL_DRAW_FRAMEBUFFER);
}

void CannyPrv::process(Ogl::Image<GL_RED>& destImg, const Ogl::Image<GL_RED>& srcImg, GLfloat minThresh, GLfloat maxThresh)
{
	doProcess(destImg, srcImg, minThresh, maxThresh);
}

void CannyPrv::process(Ogl::Image<GL_RED>& destImg, const Ogl::Image<GL_LUMINANCE>& srcImg, GLfloat minThresh, GLfloat maxThresh)
{
    doProcess(destImg, srcImg, minThresh, maxThresh);
}
