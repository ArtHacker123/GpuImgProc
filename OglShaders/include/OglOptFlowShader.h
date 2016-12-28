#pragma once

#include "IOglShader.h"

namespace Ogl
{

class OptFlowShader:public Ogl::IShader
{
public:
    OptFlowShader();
    ~OptFlowShader();

    void SetGain(GLfloat gain) { mGain = gain; };
    void SetRvalue(GLfloat rvalue) { mRvalue = rvalue; };

protected:
    void ApplyParameters(GLenum tex);

protected:
    GLfloat mGain;
    GLfloat mRvalue;

    static const GLchar vsCode[];
    static const GLchar fsCode[];
};

};
