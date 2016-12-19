#pragma once

#include "IOglShader.h"

namespace Ogl
{

class RgbaGrayShader :public Ogl::IShader
{
public:
    RgbaGrayShader();
    ~RgbaGrayShader();

    GLint GetAttribLocation(const GLchar* name);

protected:
    void ApplyParameters(GLenum tex);

private:
    static GLchar vsCode[];
    static GLchar fsCode[];
};

};
