#pragma once

#include "IOglShader.h"

namespace Ogl
{

class OptFlow3Shader:public Ogl::IShader
{
public:
	OptFlow3Shader();
	~OptFlow3Shader();

	void SetGain(GLfloat gain) { mGain = gain; };
	void SetSearch(GLint search) { mSearch = search; };

protected:
	void ApplyParameters(GLenum tex);

protected:
	GLfloat mGain;
	GLint mSearch;

	static const GLchar vsCode[];
	static const GLchar fsCode[];
};

};
