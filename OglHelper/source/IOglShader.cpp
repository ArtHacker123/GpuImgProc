#include "IOglShader.h"

using namespace Ogl;

UseShader::UseShader(Ogl::IShader& shader)
    :mShader(shader)
{
    mShader.mPgm->use();
    mShader.ApplyParameters(0);
}

UseShader::~UseShader()
{
    glUseProgram(0);
}

UseShaderImage::UseShaderImage(Ogl::IShader& shader, const Ogl::IImage& img, GLenum tex)
	:mShader(shader),
     mImage(img),
	 mActTex(tex)
{
	mShader.mPgm->use();
    mImage.bind(mActTex);
	mShader.ApplyParameters(mActTex);
}

UseShaderImage::~UseShaderImage()
{
	glUseProgram(0);
	mImage.unbind(mActTex);
}

