#pragma once

#include "IOglShader.h"

namespace Ogl
{

class NmsShader:public IShader
{
public:
	NmsShader();
	~NmsShader();

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