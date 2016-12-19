#pragma once

#include "IOglShader.h"

namespace Ogl
{

class ColorShader:public IShader
{
public:
	ColorShader();
	~ColorShader();

public:
	void setColor(GLfloat r, GLfloat g, GLfloat b, GLfloat a) { mRed = r; mGreen = g; mBlue = b; mAlpha = a; };

protected:
	void ApplyParameters(GLenum tex);

private:
	GLfloat mRed;
	GLfloat mGreen;
	GLfloat mBlue;
	GLfloat mAlpha;

	static const GLchar vsCode[];
	static const GLchar fsCode[];
};

};
