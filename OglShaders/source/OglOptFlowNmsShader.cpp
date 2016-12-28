#include "OglOptFlowNmsShader.h"

using namespace Ogl;

const GLchar OptFlowNmsShader::vsCode[] = SHADER_SOURCE_CODE(
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

const GLchar OptFlowNmsShader::fsCode[] = SHADER_SOURCE_CODE(
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
	vec3 data = texture2D(tex_in, tc).rgb;
	bool flag = false;
	for (int i = 0; !flag && i < offset.length(); i++) {
		pos.x = tc.x+(offset[i].x/float(tex_size.x));
		pos.y = tc.y+(offset[i].y/float(tex_size.y));
		if (data.b <= texture2D(tex_in, pos).b)
			flag = true;
	}
	data.b = flag?0.0f:data.b;
    fragColor = (data.b > threshold)?vec4(data.r, data.g, data.b, 0.0):vec4(0.0, 0.0, 0.0, 0.0);
}
);

OptFlowNmsShader::OptFlowNmsShader()
	:mThreshold(1.0/65536.0)
{
	mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

OptFlowNmsShader::~OptFlowNmsShader()
{
}

void OptFlowNmsShader::ApplyParameters(GLenum tex)
{
	GLint id = (tex - GL_TEXTURE0);
	mPgm->setUniform1i("tex_in", id);
	mPgm->setUniform1f("threshold", mThreshold);
}
