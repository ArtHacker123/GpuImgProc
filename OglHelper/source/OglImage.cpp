#include "OglImage.h"

using namespace Ogl;

Yuv420Image::Yuv420Image(GLsizei w, GLsizei h, void* pData, bool bPyramid)
    :mY(w, h, GL_R32F, GL_UNSIGNED_BYTE, pData, bPyramid),
     mU(w/2, h/2, GL_R32F, GL_UNSIGNED_BYTE, (pData==0)?0:((uint8_t*)pData+(w*h)), bPyramid),
     mV(w/2, h/2, GL_R32F, GL_UNSIGNED_BYTE, (pData==0)?0:((uint8_t*)pData+(w*h)+((w*h)>>2)), bPyramid)
{
}

Yuv420Image::~Yuv420Image()
{
}

void Yuv420Image::bind(GLenum tex) const
{
    mY.bind(tex);
    mU.bind((GLenum)(tex+1));
    mV.bind((GLenum)(tex+2));
}

void Yuv420Image::unbind(GLenum tex) const
{
    mY.unbind(tex);
    mU.unbind((GLenum)(tex+1));
    mV.unbind((GLenum)(tex+2));
}

void Yuv420Image::load(void* pData)
{
    size_t size = mY.width()*mY.height();
    uint8_t* pUdata = (uint8_t*)pData+size;
    uint8_t* pVdata = pUdata + (size >> 2);
    mY.load(pData);
    mU.load(pUdata);
    mV.load(pVdata);
}

void Yuv420Image::load(void* pData, GLint x, GLint y, GLsizei w, GLsizei h)
{
    size_t size = w*h;
    uint8_t* pUdata = (uint8_t*)pData + size;
    uint8_t* pVdata = pUdata + (size >> 2);
    mY.load(pData, x, y, w, h);
    mU.load(pUdata, x>>1, y>>1, w>>1, h>>1);
    mV.load(pVdata, x>>1, y>>1, w>>1, h>>1);
}

Nv12Image::Nv12Image(GLsizei w, GLsizei h, void* pData, bool bPyramid)
    :mY(w, h, GL_RED, GL_UNSIGNED_BYTE, pData, bPyramid),
     mUV(w, h/2, GL_RG, GL_UNSIGNED_BYTE, (pData==0)?0:((uint8_t*)pData+(w*h)), bPyramid)
{
}

Nv12Image::~Nv12Image()
{
}

void Nv12Image::bind(GLenum tex) const
{
    mY.bind(tex);
    mUV.bind((GLenum)(tex+1));
}

void Nv12Image::unbind(GLenum tex) const
{
    mY.unbind(tex);
    mUV.unbind((GLenum)(tex+1));
}

void Nv12Image::load(void* pData)
{
    mY.load(pData);
    mUV.load((uint8_t*)pData+(mY.width()*mY.height()));
}
