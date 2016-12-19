#include "OglYuv420HsvShader.h"

using namespace Ogl;

GLchar Yuv420HsvShader::vsCode[] = SHADER_SOURCE_CODE(
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

GLchar Yuv420HsvShader::fsCode[] = SHADER_SOURCE_CODE(
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
	vec3 hsv;
	vec3 yuv = vec3(texture(tex_y, tc).r, texture(tex_u, tc).r, texture(tex_v, tc).r);
	yuv -= vec3(1.0/16.0, 0.5, 0.5);
	vec3 color = bt709*yuv;
	float cmax = (color.r > color.g) ? color.r : color.g;
	if (color.b > cmax) cmax = color.b;
	float cmin = (color.r < color.g) ? color.r : color.g;
	if (color.b < cmin) cmin = color.b;
	float delta = cmax - cmin;
	if (delta == 0.0)
	{
		hsv.r = 0.0;
	}
	else if (cmax == color.r)
	{
		hsv.r = (60.0/360.0)*mod((color.g-color.b)/delta, 6.0);
	}
	else if (cmax == color.g)
	{
		hsv.r = (60.0/360.0)*(2.0+((color.b-color.r)/delta));
	}
	else if (cmax == color.b)
	{
		hsv.r = (60.0/360.0)*(4.0+((color.r-color.g)/delta));
	}

	if (cmax == 0.0)
	{
		hsv.g = 0.0;
	}
	else
	{
		hsv.g = delta/cmax;
	}
	hsv.b = cmax;
	fragColor = vec4(hsv.r, hsv.g, hsv.b, 1.0);
	//fragColor = vec4(hsv.r, hsv.r, hsv.r, 1.0);
}
);

Yuv420HsvShader::Yuv420HsvShader()
{
	mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

Yuv420HsvShader::~Yuv420HsvShader()
{
}

void Yuv420HsvShader::ApplyParameters(GLenum tex)
{
	GLint id = (tex - GL_TEXTURE0);
	mPgm->setUniform1i("tex_y", id);
	mPgm->setUniform1i("tex_u", id+1);
	mPgm->setUniform1i("tex_v", id+2);
}

GLint Yuv420HsvShader::GetAttribLocation(const GLchar* name)
{
	return mPgm->GetAttribLocation(name);
}
