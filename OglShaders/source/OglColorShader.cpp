#include "OglColorShader.h"

using namespace Ogl;

const GLchar ColorShader::vsCode[] = SHADER_SOURCE_CODE(
#version 150\n
in vec4 inVc;
void main(void)
{
	gl_Position = inVc;
}
);

const GLchar ColorShader::fsCode[] = SHADER_SOURCE_CODE(
#version 150\n
out vec4 fragColor;
uniform vec4 color;
void main(void)
{
	fragColor = color;
}
);

ColorShader::ColorShader()
	:mRed(1.0),
	 mGreen(0.0),
	 mBlue(0.0),
	 mAlpha(1.0)
{
	mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

ColorShader::~ColorShader()
{
}

void ColorShader::ApplyParameters(GLenum tex)
{
	mPgm->setUniform4f("color", mRed, mGreen, mBlue, mAlpha);
}
