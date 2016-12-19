#pragma once

#include "IOglShader.h"

namespace Ogl
{

class EdgeShader:public IShader
{
public:
	EdgeShader();
	~EdgeShader();

protected:
	void ApplyParameters(GLenum tex);

protected:
	static const GLchar vsCode[];
	static const GLchar fsCode[];
};

};