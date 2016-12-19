#include "OglImageFormat.h"
#include "OglImageConverter.h"

using namespace Ogl;

std::unique_ptr<Ogl::ImgRgbaGrayConverter> ImageFormat::mRgbaGray;
std::unique_ptr<Ogl::ImgRgbaRgbaConverter> ImageFormat::mRgbaRgba;
std::unique_ptr<Ogl::ImgYuv420RgbaConverter> ImageFormat::mYuv420Rgba;

void ImageFormat::convert(Ogl::Image<GL_RED>& destImg, const Ogl::Image<GL_BGR>& srcImg)
{
    if (mRgbaGray.get() == 0)
    {
        mRgbaGray.reset(new Ogl::ImgRgbaGrayConverter);
    }
    mRgbaGray->convert(destImg, srcImg);
}

void ImageFormat::convert(Ogl::Image<GL_RGBA>& destImg, const Ogl::Image<GL_BGR>& srcImg)
{
    if (mRgbaRgba.get() == 0)
    {
        mRgbaRgba.reset(new Ogl::ImgRgbaRgbaConverter);
    }
    mRgbaRgba->convert(destImg, srcImg);
}

void ImageFormat::convert(Ogl::Image<GL_RGBA>& destImg, const Ogl::Yuv420Image& srcImg)
{
    if (mYuv420Rgba.get() == 0)
    {
        mYuv420Rgba.reset(new Ogl::ImgYuv420RgbaConverter);
    }
    mYuv420Rgba->convert(destImg, srcImg);
}
