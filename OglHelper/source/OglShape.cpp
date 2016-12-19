#include "OglShape.h"

using namespace Ogl;

Shape::Shape(GLuint vcIndex, GLuint tcIndex, const std::vector<GLushort>& index, const std::vector<Ogl::GeometryCoord>& coords)
	:mVcIndex(vcIndex),
	 mTcIndex(tcIndex),
	 mIndexBuffer(0),
	 mVertexBuffer(0),
	 mVertexArray(0),
	 mIndexSize((GLsizei)index.size())
{
	glGenBuffers(1, &mIndexBuffer);
	glGenBuffers(1, &mVertexBuffer);
	glGenVertexArrays(1, &mVertexArray);

	glBindVertexArray(mVertexArray);
	glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);

	glEnableVertexAttribArray(mVcIndex);
	glVertexAttribPointer(mVcIndex, 3, GL_FLOAT, GL_FALSE, sizeof(Ogl::GeometryCoord), 0);

	glEnableVertexAttribArray(mTcIndex);
	glVertexAttribPointer(mTcIndex, 2, GL_FLOAT, GL_FALSE, sizeof(GeometryCoord), (void *)sizeof(VertexCoord));

	glBufferData(GL_ARRAY_BUFFER, sizeof(GeometryCoord)*coords.size(), coords.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort)*mIndexSize, index.data(), GL_STATIC_DRAW);

	glBindVertexArray(0);

	glDisableVertexAttribArray(mVcIndex);
	glDisableVertexAttribArray(mTcIndex);
}

Shape::~Shape()
{
	glDeleteBuffers(1, &mIndexBuffer);
	glDeleteBuffers(1, &mVertexBuffer);
	glDeleteVertexArrays(1, &mVertexArray);
}

void Shape::draw()
{
	glBindVertexArray(mVertexArray);
	glDrawElements(GL_TRIANGLES, mIndexSize, GL_UNSIGNED_SHORT, 0);
	glBindVertexArray(0);
}

void Shape::draw(Ogl::IShader& shader, const Ogl::IImage& img)
{
    Ogl::UseShaderImage use(shader, img, GL_TEXTURE0);
	draw();
}

void Shape::draw(Ogl::IShader& shader, const Ogl::IGeometry::Rect& rect, const Ogl::IImage& img)
{
	glViewport(rect.left, rect.top, rect.width, rect.height);
    Ogl::UseShaderImage use(shader, img, GL_TEXTURE0);
	draw();
}
