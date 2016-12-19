#include "OglOptFlow2Shader.h"

using namespace Ogl;

const GLchar OptFlow2Shader::vsCode[] = SHADER_SOURCE_CODE(
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

const GLchar OptFlow2Shader::fsCode[] = SHADER_SOURCE_CODE(
#version 150\n
in vec2 tc;
out float u;
out float v;
	
uniform sampler2D tex_ix2;
uniform sampler2D tex_iy2;
uniform sampler2D tex_ixiy;
uniform sampler2D tex_ixit;
uniform sampler2D tex_iyit;
	
	
uniform vec2 offset[9] = vec2[9](vec2(+0.0, +0.0), vec2(+1.0, +0.0), vec2(+2.0, +0.0),
									vec2(+0.0, +1.0), vec2(+1.0, +1.0), vec2(+2.0, +1.0),
		                            vec2(+0.0, +2.0), vec2(+1.0, +2.0), vec2(+2.0, +2.0));
	
void main(void)
{
	vec2 X = vec2(0.0, 0.0);
	mat2 A = mat2(0.0, 0.0, 0.0, 0.0);
	ivec2 tex_size = textureSize(tex_ix2, 0);
	for (int i = 0; i < offset.length(); i++) {
	    vec2 pos = tc+vec2(offset[i].x/float(tex_size.x), offset[i].y/float(tex_size.y));
	    A[0][0] += texture2D(tex_ix2, pos).r;
	    A[1][1] += texture2D(tex_iy2, pos).r;
	    A[0][1] += texture2D(tex_ixiy, pos).r;
	    X.x -= texture2D(tex_ixit, pos).r;
	    X.y -= texture2D(tex_iyit, pos).r;
	}
	A[1][0] = A[0][1];
	vec2 uv = inverse(A)*X;
	u = uv.x;
	v = uv.y;
}
);

OptFlow2Shader::OptFlow2Shader()
{
	mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

OptFlow2Shader::~OptFlow2Shader()
{
}

void OptFlow2Shader::ApplyParameters(GLenum tex)
{
	GLint id = (tex - GL_TEXTURE0);
	mPgm->setUniform1i("tex_ix2", id);
	mPgm->setUniform1i("tex_iy2", id+1);
	mPgm->setUniform1i("tex_ixiy", id+2);
	mPgm->setUniform1i("tex_ixit", id+3);
	mPgm->setUniform1i("tex_iyit", id+4);
}
