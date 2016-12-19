#pragma once

#include "IOglShader.h"

namespace Ogl
{

class Yuv420Shader :public Ogl::IShader
{
public:
    Yuv420Shader();
    ~Yuv420Shader();

    GLint GetAttribLocation(const GLchar* name);

protected:
    void ApplyParameters(GLenum tex);

private:
    static GLchar vsCode[];
    static GLchar fsCode[];
};

};
