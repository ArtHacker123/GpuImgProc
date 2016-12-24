#include "OglTexture.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")

using namespace Ogl;

Texture2D::Texture2D(GLint ifmt, GLenum fmt, GLenum type, GLsizei w, GLsizei h, void* pData, bool pyramid)
    :mType(type),
     mPyramid(pyramid),
     mFormat(fmt),
     mWidth(w),
     mHeight(h),
     mTexture(0),
     mIntFormat(ifmt)
{
    glGenTextures(1, &mTexture);
    init(pData);
}

Texture2D::~Texture2D()
{
    glDeleteTextures(1, &mTexture);
}

void Texture2D::init(void* pData)
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (mPyramid == false)?GL_LINEAR:GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, mIntFormat, mWidth, mHeight, 0, mFormat, mType, pData);
    if (mPyramid)
    {
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture2D::bind(GLenum tex) const
{
    glActiveTexture(tex);
    glBindTexture(GL_TEXTURE_2D, mTexture);
}

void Texture2D::unbind(GLenum tex)
{
    glActiveTexture(tex);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture2D::load(void* pData)
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, mFormat, mType, pData);
    if (mPyramid)
    {
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}
