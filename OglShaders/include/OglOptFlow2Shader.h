#pragma once

#include "IOglShader.h"

namespace Ogl
{

class OptFlow2Shader:public Ogl::IShader
{
public:
	OptFlow2Shader();
	~OptFlow2Shader();

protected:
	void ApplyParameters(GLenum tex);

protected:
	static const GLchar vsCode[];
	static const GLchar fsCode[];
};

};