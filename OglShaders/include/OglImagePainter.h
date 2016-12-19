#pragma once

#include <memory>

#include "OglShape.h"
#include "IOglImage.h"
#include "IOglGeometry.h"
#include "OglUvShader.h"
#include "OglLumaShader.h"
#include "OglRgbaShader.h"
#include "OglYuv420Shader.h"
#include "OglRgbaGrayShader.h"
#include "OglYuv420HsvShader.h"

namespace Ogl
{

template <class TShader, class TImage>
class ImagePainter
{
public:
	ImagePainter();
	~ImagePainter();

	void draw(const TImage& img);
	void draw(const Ogl::IGeometry::Rect& viewPort, const TImage& img);

private:
	std::unique_ptr<Ogl::Shape> mShape;
	std::unique_ptr<TShader> mShader;
};

//TODO: Fix this
extern const std::vector<GLushort> index;
extern const std::vector<Ogl::GeometryCoord> coords;

template <class TShader, class TImage>
ImagePainter<TShader, TImage>::ImagePainter()
{
	mShader.reset(new TShader);
	GLint vcIndex = mShader->GetAttribLocation("inVc");
	GLint tcIndex = mShader->GetAttribLocation("inTc");
	mShape.reset(new Ogl::Shape(vcIndex, tcIndex, index, coords));
}

template <class TShader, class TImage>
ImagePainter<TShader, TImage>::~ImagePainter()
{
}

template <class TShader, class TImage>
void ImagePainter<TShader, TImage>::draw(const TImage& img)
{
	mShape->draw(*mShader, img);
}

template <class TShader, class TImage>
void ImagePainter<TShader, TImage>::draw(const Ogl::IGeometry::Rect& viewPort, const TImage& img)
{
	GLint params[4];
	glGetIntegerv(GL_VIEWPORT, params);
	mShape->draw(*mShader, viewPort, img);
	glViewport(params[0], params[1], params[2], params[3]);
}

};
