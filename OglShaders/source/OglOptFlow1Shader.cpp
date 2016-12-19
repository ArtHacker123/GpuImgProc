#include "OglOptFlow1Shader.h"

using namespace Ogl;

const GLchar OptFlow1Shader::vsCode[] = SHADER_SOURCE_CODE(
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

const GLchar OptFlow1Shader::fsCode[] = SHADER_SOURCE_CODE(
#version 150\n
in vec2 tc;
out float ix2;
out float iy2;
out float ix_iy;
out float ix_it;
out float iy_it;
uniform sampler2D tex1;
uniform sampler2D tex2;
uniform float coeffs_fx[4] = float[4](-1.0f, +1.0f,
	                                    -1.0f, +1.0f);
uniform float coeffs_fy[4] = float[4](-1.0f, -1.0f,
	                                    +1.0f, +1.0f);
uniform vec2 offset[4] = vec2[4](vec2(0.0f, 0.0f), vec2(+1.0f, 0.0f), 
	                                vec2(0.0f, 1.0f), vec2(+1.0f, 1.0f));
void main(void)
{
	vec2 pos;
	float ix = 0.0;
	float iy = 0.0;
	float it = 0.0;
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
	ix2 = ix*ix*0.25;
	iy2 = iy*iy*0.25;
	ix_iy = ix*iy*0.25;
	ix_it = ix*it*0.5;
	iy_it = iy*it*0.5;
}
);

OptFlow1Shader::OptFlow1Shader()
{
	mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

OptFlow1Shader::~OptFlow1Shader()
{
}

void OptFlow1Shader::ApplyParameters(GLenum tex)
{
	GLint id = (tex - GL_TEXTURE0);
	mPgm->setUniform1i("tex1", id);
	mPgm->setUniform1i("tex2", id+1);
}
