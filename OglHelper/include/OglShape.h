#pragma once

#include "IOglGeometry.h"
#include "GeometryCoord.h"

#include <vector>

namespace Ogl
{

class Shape:public IGeometry
{
public:
	Shape(GLuint vcIndex, GLuint tcIndex, const std::vector<GLushort>& index, const std::vector<Ogl::GeometryCoord>& coords);
	~Shape();

public:
	void draw();
	void draw(GLenum tex);
    void draw(Ogl::IShader& shader, const Ogl::IImage& tex);
	void draw(Ogl::IShader& shader, const Ogl::IGeometry::Rect& rect, const Ogl::IImage& tex);

private:
	GLuint mVcIndex;
	GLuint mTcIndex;
	GLuint mIndexBuffer;
	GLuint mVertexBuffer;
	GLuint mVertexArray;

	GLsizei mIndexSize;
};

};
