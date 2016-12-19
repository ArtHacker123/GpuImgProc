#pragma once

#include "IOglShader.h"

namespace Ogl
{

class EigenShader:public IShader
{
public:
	EigenShader();
	~EigenShader();

protected:
	void ApplyParameters(GLenum tex);

protected:
	static const GLchar vsCode[];
	static const GLchar fsCode[];
};

};
