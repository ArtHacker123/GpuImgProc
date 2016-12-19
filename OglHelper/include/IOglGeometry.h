#pragma once

#include "IOglImage.h"
#include "IOglShader.h"

#include <memory>

namespace Ogl
{

class IGeometry
{
public:
	struct Rect
	{
		int top;
		int left;
		int width;
		int height;
	};

	virtual ~IGeometry() {};

public:
	virtual void draw(Ogl::IShader& shader, const Ogl::IImage& img) = 0;
	virtual void draw(Ogl::IShader& shader, const Rect& rect, const Ogl::IImage& img) = 0;

};

};
