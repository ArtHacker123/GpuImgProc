#pragma once

#include "IOglShader.h"

namespace Ogl
{

class OptFlowNmsShader:public IShader
{
public:
    OptFlowNmsShader();
	~OptFlowNmsShader();

	float threshold() const { return mThreshold; };
	void setThreshold(float thr) { mThreshold = thr; };

protected:
	void ApplyParameters(GLenum tex);

protected:
	float mThreshold;

	static const GLchar vsCode[];
	static const GLchar fsCode[];
};

};