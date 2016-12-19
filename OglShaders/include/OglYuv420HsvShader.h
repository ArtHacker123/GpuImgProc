#pragma once

#include "IOglShader.h"

namespace Ogl
{

class Yuv420HsvShader:public Ogl::IShader
{
public:
    Yuv420HsvShader();
    ~Yuv420HsvShader();

    GLint GetAttribLocation(const GLchar* name);

protected:
    void ApplyParameters(GLenum tex);

private:
    static GLchar vsCode[];
    static GLchar fsCode[];
};

};
