#include "OglUvShader.h"

using namespace Ogl;

GLchar UvShader::vsCode[] = SHADER_SOURCE_CODE(
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

GLchar UvShader::fsCode[] = SHADER_SOURCE_CODE(
	#version 150\n
	in vec2 tc;
	out vec4 fragColor;
	uniform sampler2D tex;
	
	void main(void)
	{
		vec2 texSize = vec2(textureSize(tex, 0));
		vec2 coord = (vec2(5.0, 5.0)+(10.0*floor(tc*texSize/10.0)))/texSize;
		vec2 uv_ofs = ceil((tc-coord)*texSize);
	    float u = texture(tex, coord).r;
	    float v = texture(tex, coord).g;
	    float color = sqrt((u*u)+(v*v));
		if (color < 1.0f) discard;
		float vu_angle = atan(v, u);
		float angle = atan(uv_ofs.y, uv_ofs.x);
		if (abs(vu_angle-angle) <= 0.1)
		   fragColor = vec4(1.0, 0.0, 0.0, 1.0);
		else
	       discard;
	}
);

UvShader::UvShader()
{
	mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

UvShader::~UvShader()
{
}

void UvShader::ApplyParameters(GLenum tex)
{
	mPgm->setUniform1i("tex", (tex - GL_TEXTURE0));
}
