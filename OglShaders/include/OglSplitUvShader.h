#pragma once

#include "IOglShader.h"

namespace Ogl
{

class SplitUvShader:public Ogl::IShader
{
public:
	SplitUvShader();
	~SplitUvShader();

protected:
	void ApplyParameters(GLenum tex);

private:
	static GLchar vsCode[];
	static GLchar fsCode[];
};

};
