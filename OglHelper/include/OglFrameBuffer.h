#pragma once

#include "OglTexture.h"

namespace Ogl
{

class FrameBuffer
{
public:
    FrameBuffer();
    ~FrameBuffer();

    void bind(GLenum target);
    static void release(GLenum target);

private:
    GLuint mFb;
};

};
