#include "OglNmesShader.h"

using namespace Ogl;

const GLchar NmesShader::vsCode[] = SHADER_SOURCE_CODE(
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

const GLchar NmesShader::fsCode[] = SHADER_SOURCE_CODE(
#version 150\n
#define ne_dir 0\n
#define n_dir  1\n
#define nw_dir 2\n
#define e_dir  3\n
#define w_dir  5\n
#define se_dir 6\n
#define s_dir  7\n
#define sw_dir 8\n
in vec2 tc;
out float fragColor;
uniform sampler2D angle;
uniform sampler2D tex_in;
uniform int first[]=int[](e_dir, se_dir, n_dir, ne_dir);
uniform int second[]=int[](w_dir, nw_dir, s_dir, sw_dir);
uniform vec2 offset[9] =
	vec2[9](vec2(-1.0f, +1.0f), vec2(+0.0f, +1.0f), vec2(+1.0f, +1.0f),
	        vec2(-1.0f, +0.0f), vec2(+0.0f, +0.0f), vec2(+1.0f, +0.0f),
	        vec2(-1.0f, -1.0f), vec2(+0.0f, -1.0f), vec2(+1.0f, -1.0f));
void main(void)
{
	vec2 pos1;
	vec2 pos2;
	ivec2 tex_size = textureSize(tex_in, 0);
	fragColor = texture2D(tex_in, tc).r;
	float angle = texture2D(angle, tc).r;
	int i = int(ceil(4.0*angle));
	pos1.x = tc.x+(offset[first[i]].x/float(tex_size.x));
	pos1.y = tc.y+(offset[first[i]].y/float(tex_size.y));
	pos2.x = tc.x+(offset[second[i]].x/float(tex_size.x));
	pos2.y = tc.y+(offset[second[i]].y/float(tex_size.y));
	if ((fragColor <= texture2D(tex_in, pos1).r) || (fragColor <= texture2D(tex_in, pos2).r))
	    fragColor = 0.0f;
}
);

NmesShader::NmesShader()
{
	mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

NmesShader::~NmesShader()
{
}

void NmesShader::ApplyParameters(GLenum tex)
{
	GLint id = (tex - GL_TEXTURE0);
	mPgm->setUniform1i("angle", id);
	mPgm->setUniform1i("tex_in", id+1);
}
