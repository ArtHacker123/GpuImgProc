#pragma once

#include "IOglShader.h"

namespace Ogl
{

class IxIyItShader:public Ogl::IShader
{
public:
	IxIyItShader();
	~IxIyItShader();

protected:
	void ApplyParameters(GLenum tex);

protected:
	static const GLchar vsCode[];
	static const GLchar fsCode[];
};

};
