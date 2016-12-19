#pragma once

#include "IOglShader.h"

namespace Ogl
{

class GaussShader:public IShader
{
public:
    GaussShader();
    ~GaussShader();

protected:
    void ApplyParameters(GLenum tex);

protected:
    static const GLchar vsCode[];
    static const GLchar fsCode[];
};

};
