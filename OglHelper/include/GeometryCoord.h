#pragma once

#include <GL/GL.h>

namespace Ogl
{

struct VertexCoord
{
    GLfloat x;
    GLfloat y;
    GLfloat z;
};

struct TextureCoord
{
    GLfloat x;
    GLfloat y;
};

struct GeometryCoord
{
    VertexCoord vc;
    TextureCoord tc;
};

};
