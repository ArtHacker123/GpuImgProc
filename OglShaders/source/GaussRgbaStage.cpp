#include "GaussRgbaStage.h"

const GLchar GaussRgbaStage::vsCode[] =
	"#version 150\n"
	"in vec4 inVc;\n"
	"in vec2 inTc;\n"
	"out vec2 tc;\n"
	"void main(void)\n"
	"{\n"
	"    gl_Position = inVc;\n"
	"    tc = inTc;\n"
	"}\n";

const GLchar GaussRgbaStage::fsCode[] =
	"#version 150\n"
	"in vec2 tc;\n"
	"out vec4 fragColor;\n"
	"uniform sampler2D tex_in;\n"
	"\n"
	"uniform float coeffs[25] = float[25]((2.0 / 159.0), (4.0 / 159.0), (5.0 / 159.0), (4.0 / 159.0), (2.0 / 159.0),\n"
	"	(4.0 / 159.0), (9.0 / 159.0), (12.0 / 159.0), (9.0 / 159.0), (4.0 / 159.0),\n"
	"	(5.0 / 159.0), (12.0 / 159.0), (15.0 / 159.0), (12.0 / 159.0), (5.0 / 159.0),\n"
	"	(4.0 / 159.0), (9.0 / 159.0), (12.0 / 159.0), (9.0 / 159.0), (4.0 / 159.0),\n"
	"	(2.0 / 159.0), (4.0 / 159.0), (5.0 / 159.0), (4.0 / 159.0), (2.0 / 159.0));\n"
	"\n"
	"uniform vec2 coord[25] = vec2[25](vec2(-2.0, +2.0), vec2(-1.0, +2.0), vec2(+0.0, +2.0), vec2(+1.0, +2.0), vec2(+2.0, +2.0),\n"
	"	vec2(-2.0, +1.0), vec2(-1.0, +1.0), vec2(+0.0, +1.0), vec2(+1.0, +1.0), vec2(+2.0, +1.0),\n"
	"	vec2(-2.0, +0.0), vec2(-1.0, +0.0), vec2(+0.0, +0.0), vec2(+1.0, +0.0), vec2(+2.0, +0.0),\n"
	"	vec2(-2.0, -1.0), vec2(-1.0, -1.0), vec2(+0.0, -1.0), vec2(+1.0, -1.0), vec2(+2.0, -1.0),\n"
	"	vec2(-2.0, -2.0), vec2(-1.0, -2.0), vec2(+0.0, -2.0), vec2(+1.0, -2.0), vec2(+2.0, -2.0));\n"
	"\n"
	"void main(void)\n"
	"{\n"
	"	vec2 pos = tc;\n"
	"	float r = 0.0, g = 0.0, b = 0.0;\n"
	"	ivec2 tex_size = textureSize(tex_in, 0);\n"
	"	for (int i = 0; i < coord.length(); i++) {\n"
	"		pos.x = tc.x + (coord[i].x / float(tex_size.x));\n"
	"		pos.y = tc.y + (coord[i].y / float(tex_size.y));\n"
	"		r += (coeffs[i] * texture(tex_in, pos).r);\n"
	"		g += (coeffs[i] * texture(tex_in, pos).g);\n"
	"		b += (coeffs[i] * texture(tex_in, pos).b);\n"
	"	}\n"
	"	fragColor = vec4(r, g, b, 0.0);\n"
	"}\n";

GaussRgbaStage::GaussRgbaStage()
	:AbsStage(1)
{
	mPgm.reset(new GlProgram(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

GaussRgbaStage::~GaussRgbaStage()
{
}

void GaussRgbaStage::process()
{
	GlUseProgram use(*mPgm);

	mPgm->setUniform1i("tex_in", 0);

	GLint tcIndex = mPgm->GetAttribLocation("inTc");
	GLint vcIndex = mPgm->GetAttribLocation("inVc");

	renderQuad(vcIndex, tcIndex, mTex);

	for (size_t i = 0; i < mTex.size(); i++)
	{
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_2D, mTex[i]->mTexture);
	}
}
