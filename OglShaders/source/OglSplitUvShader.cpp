#include "OglSplitUvShader.h"

using namespace Ogl;

GLchar SplitUvShader::vsCode[] = SHADER_SOURCE_CODE(
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

GLchar SplitUvShader::fsCode[] = SHADER_SOURCE_CODE(
#version 150\n
in vec2 tc;
out float u;
out float v;
uniform sampler2D tex_uv;
	
void main(void)
{
	u = texture(tex_uv, tc).r;
	v = texture(tex_uv, tc).g;
}
);

SplitUvShader::SplitUvShader()
{
	mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

SplitUvShader::~SplitUvShader()
{
}

void SplitUvShader::ApplyParameters(GLenum tex)
{
	mPgm->setUniform1i("tex_uv", (tex - GL_TEXTURE0));
}
