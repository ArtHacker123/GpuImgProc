#include "OglFrameBuffer.h"

using namespace Ogl;

FrameBuffer::FrameBuffer()
	:mFb(0)
{
	glGenFramebuffers(1, &mFb);
}

FrameBuffer::~FrameBuffer()
{
	glDeleteFramebuffers(1, &mFb);
}

void FrameBuffer::bind(GLenum target)
{
	glBindFramebuffer(target, mFb);
}

void FrameBuffer::release(GLenum target)
{
	glBindFramebuffer(target, 0);
}
