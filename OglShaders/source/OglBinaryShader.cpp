#include "OglBinaryShader.h"

using namespace Ogl;

const GLchar BinaryShader::vsCode[] = SHADER_SOURCE_CODE(
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

const GLchar BinaryShader::fsCode[] = SHADER_SOURCE_CODE(
#version 150\n
in vec2 tc;
out float fragColor;
uniform float min;
uniform float max;
uniform sampler2D tex_in;
uniform vec2 coord[8] = vec2[8](vec2(-1.0f, +1.0f), vec2(+0.0f, +1.0f), vec2(+1.0f, +1.0f),
			                    vec2(-1.0f, +0.0f), vec2(+1.0f, +0.0f), vec2(-1.0f, -1.0f),
			                    vec2(+0.0f, -1.0f), vec2(+1.0f, -1.0f));
void main(void)
{
	ivec2 tex_size = textureSize(tex_in, 0);
	float color = texture2D(tex_in, tc).r;
	if (color >= max)
		color = 1.0;
	else if (color >= min) {
		int i;
		vec2 pos;
		bool flag = true;
		for (i = 0; i < 8 && flag; i++) {
			pos.x = tc.x+(coord[i].x/float(tex_size.x));
			pos.y = tc.y+(coord[i].y/float(tex_size.y));
			if (max >= texture2D(tex_in, pos).r)
				flag = false;
		}
		color = (flag)?0.0f:1.0f;
	} else
		color = 0.0f;
	fragColor = color;
}
);

BinaryShader::BinaryShader()
	:minThresh(20.0/256.0),
	 maxThresh(70.0/256.0)
{
	mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

BinaryShader::~BinaryShader()
{
}

void BinaryShader::ApplyParameters(GLenum tex)
{
    GLint id = (tex-GL_TEXTURE0);
	mPgm->setUniform1i("tex_in", id);
	mPgm->setUniform1f("min", minThresh);
	mPgm->setUniform1f("max", maxThresh);
}
