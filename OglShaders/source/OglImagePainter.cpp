#include "OglImagePainter.h"

const std::vector<Ogl::GeometryCoord> Ogl::coords =
{
	{ { -1.0f, +1.0f, +0.0f }, { +0.0f, +0.0f } },
	{ { +1.0f, +1.0f, +0.0f }, { +1.0f, +0.0f } },
	{ { +1.0f, -1.0f, +0.0f }, { +1.0f, +1.0f } },
	{ { -1.0f, -1.0f, +0.0f }, { +0.0f, +1.0f } }
};

const std::vector<GLushort> Ogl::index = { 0, 1, 2, 2, 3, 0 };

