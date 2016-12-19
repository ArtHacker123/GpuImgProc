#pragma once

#include "AbsStage.h"

class GaussRgbaStage :public AbsStage
{
public:
	GaussRgbaStage();
	~GaussRgbaStage();

protected:
	void process();

protected:
	static const GLchar vsCode[];
	static const GLchar fsCode[];
};
