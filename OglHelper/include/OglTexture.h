#pragma once

#include "glew.h"
#include <GL/GL.h>

#include <cstdint>

namespace Ogl
{

class Texture2D
{
public:
    Texture2D(GLint ifmt, GLenum fmt, GLenum type, GLsizei w, GLsizei h, void* pData = 0, bool pyramid = false);
    ~Texture2D();

    void load(void* pData);
    void load(void* pData, GLint x, GLint y, GLsizei w, GLsizei h);
    void bind(GLenum tex) const;
    static void unbind(GLenum tex);

    GLsizei width() const { return mWidth; };
    GLsizei height() const { return mHeight; };

    GLenum type() const { return mType; };
    GLenum format() const { return mFormat; };
    GLint internalFormat() const { return mIntFormat; };
    bool isPyramid() const { return mPyramid; };

    GLuint texture() const { return mTexture; };

protected:
    void init(void* pData);

protected:
    GLenum mType;
    bool mPyramid;
    GLenum mFormat;
    GLsizei mWidth;
    GLsizei mHeight;
    GLuint mTexture;
    GLint mIntFormat;
};

};
