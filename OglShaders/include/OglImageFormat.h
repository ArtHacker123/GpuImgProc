#pragma once

#include "OglImage.h"
#include <memory>

namespace Ogl
{

struct ImgRgbaGrayConverter;
struct ImgRgbaRgbaConverter;
struct ImgYuv420RgbaConverter;

class ImageFormat
{
public:
    static void convert(Ogl::Image<GL_RED>& destImg, const Ogl::Image<GL_BGR>& srcImg);
    static void convert(Ogl::Image<GL_RGBA>& destImg, const Ogl::Image<GL_BGR>& srcImg);
    static void convert(Ogl::Image<GL_RGBA>& destImg, const Ogl::Yuv420Image& srcImg);

private:
    ImageFormat() {};
    ~ImageFormat() {};

private:
    static std::unique_ptr<Ogl::ImgRgbaGrayConverter> mRgbaGray;
    static std::unique_ptr<Ogl::ImgRgbaRgbaConverter> mRgbaRgba;
    static std::unique_ptr<Ogl::ImgYuv420RgbaConverter> mYuv420Rgba;
};

};
