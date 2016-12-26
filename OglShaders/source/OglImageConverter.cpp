#include "OglImageConverter.h"

extern const std::vector<GLushort> index;

using namespace Ogl;

#include "OglFrameBuffer.h"

const std::vector<Ogl::GeometryCoord> Ogl::invCoords =
{
    { { -1.0f, +1.0f, +0.0f },{ +0.0f, +1.0f } },
    { { +1.0f, +1.0f, +0.0f },{ +1.0f, +1.0f } },
    { { +1.0f, -1.0f, +0.0f },{ +1.0f, +0.0f } },
    { { -1.0f, -1.0f, +0.0f },{ +0.0f, +0.0f } }
};

const GLenum Ogl::mBuffs[] =
{
    GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2,
    GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5
};

ImgRgbaGrayConverter::ImgRgbaGrayConverter()
{
    GLint vcIndex = mRgbaGrayShader.GetAttribLocation("inVc");
    GLint tcIndex = mRgbaGrayShader.GetAttribLocation("inTc");
    mRect.reset(new Ogl::Shape(vcIndex, tcIndex, index, invCoords));
}

ImgRgbaGrayConverter::~ImgRgbaGrayConverter()
{
}

void ImgRgbaGrayConverter::convert(Ogl::Image<GL_RED>& destImg, const Ogl::Image<GL_BGR>& srcImg)
{
    GLint params[4];
    Ogl::FrameBuffer fb;
    glGetIntegerv(GL_VIEWPORT, params);
    glViewport(0, 0, destImg.width(), destImg.height());
    fb.bind(GL_DRAW_FRAMEBUFFER);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, destImg.texture(), 0);
    mRect->draw(mRgbaGrayShader, srcImg);
    Ogl::FrameBuffer::release(GL_DRAW_FRAMEBUFFER);
    if (destImg.isPyramid())
    {
        destImg.bind(GL_TEXTURE0);
        glGenerateMipmap(GL_TEXTURE_2D);
        destImg.unbind(GL_TEXTURE0);
    }
    glViewport(params[0], params[1], params[2], params[3]);
}

void ImgRgbaGrayConverter::convert(Ogl::Image<GL_RED>& destImg, const Ogl::Image<GL_RGBA>& srcImg)
{
    GLint params[4];
    Ogl::FrameBuffer fb;
    glGetIntegerv(GL_VIEWPORT, params);
    glViewport(0, 0, destImg.width(), destImg.height());
    fb.bind(GL_DRAW_FRAMEBUFFER);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, destImg.texture(), 0);
    mRect->draw(mRgbaGrayShader, srcImg);
    Ogl::FrameBuffer::release(GL_DRAW_FRAMEBUFFER);
    glViewport(params[0], params[1], params[2], params[3]);
}

ImgRgbaRgbaConverter::ImgRgbaRgbaConverter()
{
    GLint vcIndex = mRgbaShader.GetAttribLocation("inVc");
    GLint tcIndex = mRgbaShader.GetAttribLocation("inTc");
    mRect.reset(new Ogl::Shape(vcIndex, tcIndex, index, invCoords));
}

ImgRgbaRgbaConverter::~ImgRgbaRgbaConverter()
{
}

void ImgRgbaRgbaConverter::convert(Ogl::Image<GL_RGBA>& destImg, const Ogl::Image<GL_BGR>& srcImg)
{
    GLint params[4];
    Ogl::FrameBuffer fb;
    glGetIntegerv(GL_VIEWPORT, params);
    glViewport(0, 0, destImg.width(), destImg.height());
    fb.bind(GL_DRAW_FRAMEBUFFER);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, destImg.texture(), 0);
    mRect->draw(mRgbaShader, srcImg);
    Ogl::FrameBuffer::release(GL_DRAW_FRAMEBUFFER);
    glViewport(params[0], params[1], params[2], params[3]);
}


ImgYuv420RgbaConverter::ImgYuv420RgbaConverter()
{
    GLint vcIndex = mYuvShader.GetAttribLocation("inVc");
    GLint tcIndex = mYuvShader.GetAttribLocation("inTc");
    mRect.reset(new Ogl::Shape(vcIndex, tcIndex, index, invCoords));
}

ImgYuv420RgbaConverter::~ImgYuv420RgbaConverter()
{
}

void ImgYuv420RgbaConverter::convert(Ogl::Image<GL_RGBA>& destImg, const Ogl::Yuv420Image& srcImg)
{
    GLint params[4];
    Ogl::FrameBuffer fb;
    glGetIntegerv(GL_VIEWPORT, params);
    glViewport(0, 0, destImg.width(), destImg.height());
    fb.bind(GL_DRAW_FRAMEBUFFER);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, destImg.texture(), 0);
    mRect->draw(mYuvShader, srcImg);
    Ogl::FrameBuffer::release(GL_DRAW_FRAMEBUFFER);
    glViewport(params[0], params[1], params[2], params[3]);
}

