#pragma once

#include "IOglShader.h"

namespace Ogl
{

class IxIyShader:public Ogl::IShader
{
public:
	IxIyShader();
	~IxIyShader();

protected:
	void ApplyParameters(GLenum tex);

protected:
	static const GLchar vsCode[];
	static const GLchar fsCode[];
};

};
