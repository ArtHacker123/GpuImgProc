#pragma once

#include "IOglShader.h"

namespace Ogl
{

class RgbaShader:public Ogl::IShader
{
public:
	RgbaShader();
	~RgbaShader();

	GLint GetAttribLocation(const GLchar* name);

protected:
	void ApplyParameters(GLenum tex);

private:
	static GLchar vsCode[];
	static GLchar fsCode[];
};

};
