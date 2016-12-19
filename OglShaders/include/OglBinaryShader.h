#pragma once

#include "IOglShader.h"

namespace Ogl
{

class BinaryShader:public IShader
{
public:
	BinaryShader();
	~BinaryShader();

public:
	float minThreshold() const { return minThresh; };
	float maxThreshold() const { return maxThresh; };

	void setMinThreshold(float min) { minThresh = min; };
	void setMaxThreshold(float max) { maxThresh = max; };

protected:
	void ApplyParameters(GLenum tex);

private:
	float minThresh;
	float maxThresh;

	static const GLchar vsCode[];
	static const GLchar fsCode[];
};

};
