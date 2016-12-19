#pragma once

#include "OglImage.h"
#include "OglImagePainter.h"

class GlView
{
public:
	enum KEYS { LEFT, RIGHT, UP, DOWN };

	GlView(GLsizei w, GLsizei h);
	~GlView();

public:
	void draw(uint8_t* pData);
	void resize(GLsizei w, GLsizei h);

	void keyDown(KEYS key);

private:
    GLfloat minThresh;
    GLfloat maxThresh;

	Ogl::Yuv420Image mYuvImg;
	Ogl::Image<GL_RED> mEdgeImg;

	Ogl::ImagePainter< Ogl::Yuv420Shader, Ogl::Yuv420Image > mYuvPainter;
	Ogl::ImagePainter< Ogl::LumaShader, Ogl::Image<GL_RED> > mLumaPainter;
};
