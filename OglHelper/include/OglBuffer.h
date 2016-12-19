#pragma once

#include "glew.h"
#include <GL/GL.h>

#include <cstdint>

namespace Ogl
{

class Buffer
{
public:
    Buffer(GLenum target, GLsizeiptr size, const GLvoid* pData, GLenum usage);
    ~Buffer();

    GLuint buffer() const { return mBuffId; };

    void bind() const;
    void unbind() const;

protected:
    GLuint mBuffId;
    GLenum mTarget;
};

};