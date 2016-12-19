#pragma once

#include "IOglShader.h"

namespace Ogl
{

class OptFlow1Shader:public Ogl::IShader
{
public:
	OptFlow1Shader();
	~OptFlow1Shader();

protected:
	void ApplyParameters(GLenum tex);

protected:
	static const GLchar vsCode[];
	static const GLchar fsCode[];
};

};
