#pragma once

#include "IOglShader.h"

namespace Ogl
{

class NmesShader :public Ogl::IShader
{
public:
    NmesShader();
    ~NmesShader();

protected:
    void ApplyParameters(GLenum tex);

protected:
    static const GLchar vsCode[];
    static const GLchar fsCode[];
};

};