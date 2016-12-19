#pragma once

#include "OglImage.h"
#include <memory>

namespace Ogl
{

class CannyPrv;

class Canny
{
public:
    static void init();
	static void process(Ogl::Image<GL_RED>& destImg, const Ogl::Image<GL_RED>& srcImg, GLfloat minThresh, GLfloat maxThresh);
    static void process(Ogl::Image<GL_RED>& destImg, const Ogl::Image<GL_LUMINANCE>& srcImg, GLfloat minThresh, GLfloat maxThresh);

protected:
	static std::unique_ptr<Ogl::CannyPrv> mPrv;
};

};
