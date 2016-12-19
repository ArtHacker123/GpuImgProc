#include "OglMatrixMShader.h"

using namespace Ogl;

GLchar MatrixMShader::vsCode[] = SHADER_SOURCE_CODE(
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

GLchar MatrixMShader::fsCode[] = SHADER_SOURCE_CODE(
#version 150\n
in vec2 tc;
out float ix2;
out float iy2;
out float ixiy;
uniform sampler2D tex_in;
	
uniform float coeffs_fx[9] = float[9](-1.0f, +0.0f, +1.0f,
	                                -2.0f, +0.0f, +2.0f,
	                                -1.0f, +0.0f, +1.0f);
	
uniform float coeffs_fy[9] = float[9](+1.0f, +2.0f, +1.0f, 
	                                +0.0f, +0.0f, +0.0f,
	                                -1.0f, -2.0f, -1.0f);
	
uniform vec2 offset[9] = vec2[9](vec2(-1.0f, +1.0f), vec2(+0.0f, +1.0f), vec2(+1.0f, +1.0f), 
	                            vec2(-1.0f, +0.0f), vec2(+0.0f, +0.0f), vec2(+1.0f, +0.0f), 
	                            vec2(-1.0f, -1.0f), vec2(+0.0f, -1.0f), vec2(+1.0f, -1.0f));
void main(void)
{
	vec2 pos;
	float y = 0.0f;
	float ix = 0.0f;
	float iy = 0.0f;
	ivec2 tex_size = textureSize(tex_in, 0);
	for (int i = 0; i < offset.length(); i++) {
	    pos.x = tc.x+(offset[i].x/float(tex_size.x));
	    pos.y = tc.y+(offset[i].y/float(tex_size.y));
	    y = texture2D(tex_in, pos).r;
	    ix += (y*coeffs_fx[i]);
	    iy += (y*coeffs_fy[i]);
	}
	ix2 = (ix*ix);
	iy2 = (iy*iy);
	ixiy = (ix*iy);
};
);

MatrixMShader::MatrixMShader()
{
	mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

MatrixMShader::~MatrixMShader()
{
}

void MatrixMShader::ApplyParameters(GLenum tex)
{
	mPgm->setUniform1i("tex_in", (tex - GL_TEXTURE0));
}
