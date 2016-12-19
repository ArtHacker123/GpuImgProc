#pragma once

#include <memory>
#include "IOglShader.h"

namespace Ogl
{

template <class TShader>
class Painter
{
public:
	Painter();
	~Painter();

	void SetColor(const GLfloat color[4]);
	void SetColor(const GLfloat r, const GLfloat g, const GLfloat b, const GLfloat a);
	void draw(GLenum mode, GLint first, GLsizei count, const Ogl::Buffer& buffer);

private:
	std::unique_ptr<TShader> mShader;
};

template <class TShader>
Painter<TShader>::Painter()
{
	mShader.reset(new TShader);
}

template <class TShader>
Painter<TShader>::~Painter()
{
}

template <class TShader>
void Painter<TShader>::SetColor(const GLfloat color[4])
{
	mShader->setColor(color[0], color[1], color[2], color[3]);
}

template <class TShader>
void Painter<TShader>::SetColor(const GLfloat r, const GLfloat g, const GLfloat b, const GLfloat a)
{
	mShader->setColor(r, g, b, a);
}

template <class TShader>
void Painter<TShader>::draw(GLenum mode, GLint first, GLsizei count, const Ogl::Buffer& buffer)
{
	GLint vcIndex = mShader->GetAttribLocation("inVc");

	Ogl::UseShader use(*mShader);
	buffer.bind();
	glEnableVertexAttribArray(vcIndex);
	glVertexAttribPointer(vcIndex, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glDrawArrays(mode, first, count);
	glDisableVertexAttribArray(vcIndex);
	buffer.unbind();
}

};
