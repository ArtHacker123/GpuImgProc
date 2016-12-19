#include "OglCanny.h"
#include "OglCannyPrv.h"

using namespace Ogl;

std::unique_ptr<Ogl::CannyPrv> Canny::mPrv;

void Canny::init()
{
    if (mPrv.get() == 0)
    {
        mPrv.reset(new Ogl::CannyPrv);
    }
}

void Canny::process(Ogl::Image<GL_RED>& destImg, const Ogl::Image<GL_RED>& srcImg, GLfloat minThresh, GLfloat maxThresh)
{
    Ogl::Canny::init();
    mPrv->process(destImg, srcImg, minThresh, maxThresh);
}

void Canny::process(Ogl::Image<GL_RED>& destImg, const Ogl::Image<GL_LUMINANCE>& srcImg, GLfloat minThresh, GLfloat maxThresh)
{
    Ogl::Canny::init();
    mPrv->process(destImg, srcImg, minThresh, maxThresh);
}
