#include "OglBuffer.h"

using namespace Ogl;

Buffer::Buffer(GLenum target, GLsizeiptr size, const GLvoid* pData, GLenum usage)
	:mBuffId(0),
	 mTarget(target)
{
	glGenBuffers(1, &mBuffId);
	glBindBuffer(target, mBuffId);
	glBufferData(target, size, pData, usage);
	glBindBuffer(target, 0);
}

Buffer::~Buffer()
{
	glDeleteBuffers(1, &mBuffId);
}

void Buffer::bind() const
{
	glBindBuffer(mTarget, mBuffId);
}

void Buffer::unbind() const
{
	glBindBuffer(mTarget, 0);
}
