#pragma once

#include "OglShape.h"
#include "OglImage.h"
#include "OglRgbaShader.h"
#include "OglRgbaGrayShader.h"
#include "OglYuv420Shader.h"

#include <vector>
#include <memory>

namespace Ogl
{

extern const GLenum mBuffs[];
extern const std::vector<GLushort> index;
extern const std::vector<Ogl::GeometryCoord> invCoords;

struct ImgRgbaGrayConverter
{
public:
    ImgRgbaGrayConverter();
    ~ImgRgbaGrayConverter();

    void convert(Ogl::Image<GL_RED>& destImg, const Ogl::Image<GL_BGR>& srcImg);
    void convert(Ogl::Image<GL_RED>& destImg, const Ogl::Image<GL_RGBA>& srcImg);

private:
    std::unique_ptr<Ogl::Shape> mRect;
    Ogl::RgbaGrayShader mRgbaGrayShader;
};

struct ImgRgbaRgbaConverter
{
public:
    ImgRgbaRgbaConverter();
    ~ImgRgbaRgbaConverter();

    void convert(Ogl::Image<GL_RGBA>& destImg, const Ogl::Image<GL_BGR>& srcImg);

private:
    std::unique_ptr<Ogl::Shape> mRect;
    Ogl::RgbaShader mRgbaShader;
};

struct ImgYuv420RgbaConverter
{
public:
    ImgYuv420RgbaConverter();
    ~ImgYuv420RgbaConverter();

    void convert(Ogl::Image<GL_RGBA>& destImg, const Ogl::Yuv420Image& srcImg);

private:
    std::unique_ptr<Ogl::Shape> mRect;
    Ogl::Yuv420Shader mYuvShader;
};

};
