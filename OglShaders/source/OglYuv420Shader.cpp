#include "OglYuv420Shader.h"

using namespace Ogl;

GLchar Yuv420Shader::vsCode[] = SHADER_SOURCE_CODE(
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

GLchar Yuv420Shader::fsCode[] = SHADER_SOURCE_CODE(
#version 150\n
in vec2 tc;
out vec4 fragColor;
uniform sampler2D tex_y;
uniform sampler2D tex_u;
uniform sampler2D tex_v;
const mat3 bt709 = mat3(1.164,  1.164, 1.164,
						0.0,   -0.213, 2.115,
						1.789, -0.534, 0.0);
	
void main(void)
{
	vec3 yuv = vec3(texture(tex_y, tc).r, texture(tex_u, tc).r, texture(tex_v, tc).r);
	yuv -= vec3(1.0/16.0, 0.5, 0.5);
	vec3 color = bt709*yuv;
	fragColor = vec4(color.rgb, 1.0);
}
);

Yuv420Shader::Yuv420Shader()
{
	mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

Yuv420Shader::~Yuv420Shader()
{
}

void Yuv420Shader::ApplyParameters(GLenum tex)
{
	GLint id = (tex - GL_TEXTURE0);
	mPgm->setUniform1i("tex_y", id);
	mPgm->setUniform1i("tex_u", id+1);
	mPgm->setUniform1i("tex_v", id+2);
}

GLint Yuv420Shader::GetAttribLocation(const GLchar* name)
{
	return mPgm->GetAttribLocation(name);
}
