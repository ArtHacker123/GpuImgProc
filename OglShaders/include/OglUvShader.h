#pragma once

#include "IOglShader.h"

namespace Ogl
{

class UvShader:public Ogl::IShader
{
public:
	UvShader();
	~UvShader();

protected:
	void ApplyParameters(GLenum tex);

private:
	static GLchar vsCode[];
	static GLchar fsCode[];
};

};
