#include "OglNmsShader.h"

using namespace Ogl;

const GLchar NmsShader::vsCode[] = SHADER_SOURCE_CODE(
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

const GLchar NmsShader::fsCode[] = SHADER_SOURCE_CODE(
#version 150\n
in vec2 tc;
out vec4 fragColor;
uniform float threshold;
uniform sampler2D tex_in;
uniform vec2 offset[8] = vec2[8](vec2(-1.0f, +1.0f), vec2(+0.0f, +1.0f), vec2(+1.0f, +1.0f), 
									vec2(-1.0f, +0.0f), vec2(+1.0f, +0.0f), vec2(-1.0f, -1.0f), 
									vec2(+0.0f, -1.0f), vec2(+1.0f, -1.0f));
void main(void)
{
	vec2 pos;
	ivec2 tex_size = textureSize(tex_in, 0);
	float data = texture2D(tex_in, tc).r;
	bool flag = false;
	for (int i = 0; !flag && i < offset.length(); i++) {
		pos.x = tc.x+(offset[i].x/float(tex_size.x));
		pos.y = tc.y+(offset[i].y/float(tex_size.y));
		if (data <= texture2D(tex_in, pos).r)
			flag = true;
	}
	data = flag?0.0f:data;
	//data = (data > 0.0f)?1.0f:0.0f;
	data = (data > threshold)?1.0f:0.0f;
	fragColor = vec4(data, 0.0f, 0.0f, data);
}
);

NmsShader::NmsShader()
	:mThreshold(1.0/65536.0)
{
	mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

NmsShader::~NmsShader()
{
}

void NmsShader::ApplyParameters(GLenum tex)
{
	GLint id = (tex - GL_TEXTURE0);
	mPgm->setUniform1i("tex_in", id);
	mPgm->setUniform1f("threshold", mThreshold);
}
