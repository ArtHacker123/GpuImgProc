#include "OglIxIyItShader.h"

using namespace Ogl;

const GLchar IxIyItShader::vsCode[] = SHADER_SOURCE_CODE(
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

const GLchar IxIyItShader::fsCode[] = SHADER_SOURCE_CODE(
#version 150\n
in vec2 tc;
out float ix;
out float iy;
out float it;
uniform sampler2D tex1;
uniform sampler2D tex2;
uniform float coeffs_fx[4] = float[4](-1.0f, +1.0f, -1.0f, +1.0f);
uniform float coeffs_fy[4] = float[4](-1.0f, -1.0f, +1.0f, +1.0f);
uniform vec2 offset[4] = vec2[4](vec2(0.0f, 0.0f), vec2(+1.0f, 0.0f), 
	                             vec2(0.0f, 1.0f), vec2(+1.0f, 1.0f));
void main(void)
{
	vec2 pos;
	ix = 0.0;
	iy = 0.0;
	it = 0.0;
	ivec2 tex_size = textureSize(tex1, 0);
	for (int i = 0; i < offset.length(); i++) {
	    pos.x = tc.x+(offset[i].x/float(tex_size.x));
	    pos.y = tc.y+(offset[i].y/float(tex_size.y));
	    ix += (texture2D(tex1, pos).r*coeffs_fx[i]);
	    ix += (texture2D(tex2, pos).r*coeffs_fx[i]);
	    iy += (texture2D(tex1, pos).r*coeffs_fy[i]);
		iy += (texture2D(tex2, pos).r*coeffs_fy[i]);
		it += (texture2D(tex2, pos).r-texture2D(tex1, pos).r);
	}
}
);

IxIyItShader::IxIyItShader()
{
	mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

IxIyItShader::~IxIyItShader()
{
}

void IxIyItShader::ApplyParameters(GLenum tex)
{
	GLint id = (tex - GL_TEXTURE0);
	mPgm->setUniform1i("tex1", id);
	mPgm->setUniform1i("tex2", id+1);
}
