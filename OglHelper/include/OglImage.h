#pragma once

#include "IOglImage.h"

namespace Ogl
{

template <GLenum Format>
class Image:public Ogl::IImage
{
public:
    Image(GLsizei w, GLsizei h, GLint ifmt, GLenum dataType, void* pData = 0, bool bPyramid = false):mTexture(ifmt, Format, dataType, w, h, pData, bPyramid) {};
    ~Image() {};

    GLsizei width() const { return mTexture.width(); };
    GLsizei height() const { return mTexture.height(); };

    bool isPyramid() const { return mTexture.isPyramid(); };

    GLuint texture() const { return mTexture.texture(); };

    void load(void* pData) { return mTexture.load(pData); };
    void bind(GLenum tex) const { mTexture.bind(tex); };
    void unbind(GLenum tex) const { Ogl::Texture2D::unbind(tex); };

private:
    Ogl::Texture2D mTexture;
};

class Yuv420Image:public Ogl::IImage
{
public:
    Yuv420Image(GLsizei w, GLsizei h, void* pData = 0, bool bPyramid = false);
    ~Yuv420Image();

    GLsizei width() const { return mY.width(); };
    GLsizei height() const { return mY.height(); };

    const Ogl::Image<GL_RED>& yImage() { return mY; };
    const Ogl::Image<GL_RED>& uImage() { return mU; };
    const Ogl::Image<GL_RED>& vImage() { return mV; };

    void load(void* pData);
    void bind(GLenum tex) const;
    void unbind(GLenum tex) const;

protected:
    Ogl::Image<GL_RED> mY;
    Ogl::Image<GL_RED> mU;
    Ogl::Image<GL_RED> mV;
};

class Nv12Image:public Ogl::IImage
{
public:
    Nv12Image(GLsizei w, GLsizei h, void* pData = 0, bool bPyramid = false);
    ~Nv12Image();

    GLsizei width() const { return mY.width(); };
    GLsizei height() const { return mY.height(); };

    const Ogl::Image<GL_RED>& yImage() { return mY; };
    const Ogl::Image<GL_RG>& uImage() { return mUV; };

    void load(void* pData);
    void bind(GLenum tex) const;
    void unbind(GLenum tex) const;

protected:
    Ogl::Image<GL_RED> mY;
    Ogl::Image<GL_RG> mUV;
};

};


