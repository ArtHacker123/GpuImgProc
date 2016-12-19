#include "OglGaussShader.h"

using namespace Ogl;

const GLchar GaussShader::vsCode[] = SHADER_SOURCE_CODE(
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

const GLchar GaussShader::fsCode[] = SHADER_SOURCE_CODE(
#version 150\n
in vec2 tc;
out float fragColor;
uniform sampler2D tex_y;
	
uniform float coeffs[25] = float[25]((2.0 / 159.0), (4.0 / 159.0), (5.0 / 159.0), (4.0 / 159.0), (2.0 / 159.0),
	(4.0 / 159.0), (9.0 / 159.0), (12.0 / 159.0), (9.0 / 159.0), (4.0 / 159.0),
	(5.0 / 159.0), (12.0 / 159.0), (15.0 / 159.0), (12.0 / 159.0), (5.0 / 159.0),
	(4.0 / 159.0), (9.0 / 159.0), (12.0 / 159.0), (9.0 / 159.0), (4.0 / 159.0),
	(2.0 / 159.0), (4.0 / 159.0), (5.0 / 159.0), (4.0 / 159.0), (2.0 / 159.0));
	
uniform vec2 coord[25] = vec2[25](vec2(-2.0, +2.0), vec2(-1.0, +2.0), vec2(+0.0, +2.0), vec2(+1.0, +2.0), vec2(+2.0, +2.0),
	vec2(-2.0, +1.0), vec2(-1.0, +1.0), vec2(+0.0, +1.0), vec2(+1.0, +1.0), vec2(+2.0, +1.0),
	vec2(-2.0, +0.0), vec2(-1.0, +0.0), vec2(+0.0, +0.0), vec2(+1.0, +0.0), vec2(+2.0, +0.0),
	vec2(-2.0, -1.0), vec2(-1.0, -1.0), vec2(+0.0, -1.0), vec2(+1.0, -1.0), vec2(+2.0, -1.0),
	vec2(-2.0, -2.0), vec2(-1.0, -2.0), vec2(+0.0, -2.0), vec2(+1.0, -2.0), vec2(+2.0, -2.0));
	
void main(void)
{
	vec2 pos;
	fragColor = 0.0;
	ivec2 tex_size = textureSize(tex_y, 0);
	for (int i = 0; i < coord.length(); i++) {
		pos.x = tc.x + (coord[i].x/float(tex_size.x));
		pos.y = tc.y + (coord[i].y/float(tex_size.y));
		fragColor += (coeffs[i] * texture(tex_y, pos).r);
	}
}
);

GaussShader::GaussShader()
{
	mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

GaussShader::~GaussShader()
{
}

void GaussShader::ApplyParameters(GLenum tex)
{
	GLint id = (tex - GL_TEXTURE0);
	mPgm->setUniform1i("tex_y", id);
}
