#include "OglRgbaGrayShader.h"

using namespace Ogl;

GLchar RgbaGrayShader::vsCode[] = SHADER_SOURCE_CODE(
#version 150\n
in vec4 inVc;
in vec2 inTc;
out vec2 tc;
void main(void)
{
	gl_Position = inVc;
	tc = inTc;
}
);

GLchar RgbaGrayShader::fsCode[] = SHADER_SOURCE_CODE(
#version 150\n
in vec2 tc;
out float fragColor;
uniform sampler2D tex;
void main(void)
{
	fragColor = (texture(tex, tc).r + texture(tex, tc).g + texture(tex, tc).b)/3.0;
}
);

RgbaGrayShader::RgbaGrayShader()
{
	mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

RgbaGrayShader::~RgbaGrayShader()
{
}

void RgbaGrayShader::ApplyParameters(GLenum tex)
{
	GLint id = (tex - GL_TEXTURE0);
	mPgm->setUniform1i("tex", id);
}

GLint RgbaGrayShader::GetAttribLocation(const GLchar* name)
{
	return mPgm->GetAttribLocation(name);
}
