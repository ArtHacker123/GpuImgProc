#include "OglEigenShader.h"

using namespace Ogl;

const GLchar EigenShader::vsCode[] = SHADER_SOURCE_CODE(
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

const GLchar EigenShader::fsCode[] = SHADER_SOURCE_CODE(
#version 150\n
in vec2 tc;
out float fragColor;
uniform sampler2D tex_ix2;
uniform sampler2D tex_iy2;
uniform sampler2D tex_ixiy;
void main(void)
{
	float a = texture2D(tex_ix2, tc).r;
	float b = texture2D(tex_iy2, tc).r;
	float c = texture2D(tex_ixiy, tc).r;
	float detA = (a*b)-(c*c);
	float traceA = (a+b)*(a+b);
	float R = detA-(0.15*traceA);
	if (R < 0.0f) R = 0.0f;
	fragColor = R;
}
);

EigenShader::EigenShader()
{
	mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

EigenShader::~EigenShader()
{
}

void EigenShader::ApplyParameters(GLenum tex)
{
	GLint id = (tex - GL_TEXTURE0);
	mPgm->setUniform1i("tex_ix2", id);
	mPgm->setUniform1i("tex_iy2", id+1);
	mPgm->setUniform1i("tex_ixiy", id+2);
}
