#include "OglOptFlowShader.h"

using namespace Ogl;

const GLchar OptFlowShader::vsCode[] = SHADER_SOURCE_CODE(
#version 150\n
in vec4 inVc;
in vec2 inTc;
out vec2 tc;
void main(void)
{
    gl_Position = inVc;
    tc = inTc;
}
);

const GLchar OptFlowShader::fsCode[] = SHADER_SOURCE_CODE(
#version 150\n
in vec2 tc;
out vec3 uv;
    
uniform float gain = 2.0;
uniform float rvalue = 0.000000250f;
uniform sampler2D tex_puv;
uniform sampler2D tex_ix;
uniform sampler2D tex_iy;
uniform sampler2D tex_it;
uniform sampler2D tex1;
uniform sampler2D tex2;
    
uniform float coeffs[25] = float[25]((2.0/159.0), (4.0/159.0), (5.0/159.0), (4.0/159.0), (2.0/159.0),
                                     (4.0/159.0), (9.0/159.0), (12.0/159.0), (9.0/159.0), (4.0/159.0),
                                     (5.0/159.0), (12.0/159.0), (15.0/159.0), (12.0/159.0), (5.0/159.0),
                                     (4.0/159.0), (9.0/159.0), (12.0/159.0), (9.0/159.0), (4.0/159.0),
                                     (2.0/159.0), (4.0/159.0), (5.0/159.0), (4.0/159.0), (2.0/159.0));

uniform vec2 offset[25] = vec2[25](vec2(-2.0, +2.0), vec2(-1.0, +2.0), vec2(+0.0, +2.0), vec2(+1.0, +2.0), vec2(+2.0, +2.0),
                                   vec2(-2.0, +1.0), vec2(-1.0, +1.0), vec2(+0.0, +1.0), vec2(+1.0, +1.0), vec2(+2.0, +1.0),
                                   vec2(-2.0, +0.0), vec2(-1.0, +0.0), vec2(+0.0, +0.0), vec2(+1.0, +0.0), vec2(+2.0, +0.0),
                                   vec2(-2.0, -1.0), vec2(-1.0, -1.0), vec2(+0.0, -1.0), vec2(+1.0, -1.0), vec2(+2.0, -1.0),
                                   vec2(-2.0, -2.0), vec2(-1.0, -2.0), vec2(+0.0, -2.0), vec2(+1.0, -2.0), vec2(+2.0, -2.0));

uniform vec2 pos_offs[4] = vec2[4](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0));

void main(void)
{
    vec2 X = vec2(0.0, 0.0);
    mat2 A = mat2(0.0, 0.0, 0.0, 0.0);
    vec2 texSize = vec2(textureSize(tex_ix, 0));
    vec2 curr = tc*texSize;
    uv.rg = 2.0*texture2D(tex_puv, tc).rg;
    for (int i = 0; i < offset.length(); i++) {
        vec2 pos = (curr+offset[i])/texSize;
        vec2 pos1 = pos+(uv.rg/texSize);
        float ix = texture2D(tex_ix, pos).r;
        float iy = texture2D(tex_iy, pos).r;
        float it = (texture2D(tex2, pos1).r-texture2D(tex1, pos).r);
        it += (texture2D(tex2, pos1+(pos_offs[1]/texSize)).r-texture2D(tex1, pos+(pos_offs[1]/texSize)).r);
        it += (texture2D(tex2, pos1+(pos_offs[2]/texSize)).r-texture2D(tex1, pos+(pos_offs[2]/texSize)).r);
        it += (texture2D(tex2, pos1+(pos_offs[3]/texSize)).r-texture2D(tex1, pos+(pos_offs[3]/texSize)).r);
        A[0][0] += (ix*ix*coeffs[i]);
        A[1][1] += (iy*iy*coeffs[i]);
        A[0][1] += (ix*iy*coeffs[i]);
        X.x -= (ix*it*coeffs[i]);
        X.y -= (iy*it*coeffs[i]);
    }
    A[1][0] = A[0][1];
    float trace = (A[0][0] + A[1][1]);
    float R = determinant(A)-(0.04*trace*trace);
    if (R >= rvalue)
    {
        uv.rg += (inverse(A)*X);
        uv.b = length(uv.rg);
    }
    else
    {
        uv = vec3(0.0, 0.0, 0.0);
    }
}
);

OptFlowShader::OptFlowShader()
    :mGain(1.0),
     mRvalue(0.000000250f)
{
    mPgm.reset(new Ogl::Program(vsCode, sizeof(vsCode), fsCode, sizeof(fsCode)));
}

OptFlowShader::~OptFlowShader()
{
}

void OptFlowShader::ApplyParameters(GLenum tex)
{
    GLint id = (tex - GL_TEXTURE0);
    mPgm->setUniform1f("gain", mGain);
    mPgm->setUniform1f("rvalue", mRvalue);
    mPgm->setUniform1i("tex_puv", id);
    mPgm->setUniform1i("tex_ix", id+1);
    mPgm->setUniform1i("tex_iy", id+2);
    mPgm->setUniform1i("tex1", id+3);
    mPgm->setUniform1i("tex2", id+4);
}
