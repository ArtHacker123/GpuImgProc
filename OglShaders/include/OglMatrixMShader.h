#pragma once

#include "IOglShader.h"

namespace Ogl
{

class MatrixMShader:public Ogl::IShader
{
public:
	MatrixMShader();
	~MatrixMShader();

protected:
	void ApplyParameters(GLenum tex);

private:
	static GLchar fsCode[];
	static GLchar vsCode[];
};

};
