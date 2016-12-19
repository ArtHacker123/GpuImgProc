#pragma once

#include <cstdint>
#include "OglTexture.h"

namespace Ogl
{

class IImage
{
public:
    virtual ~IImage() {};

    virtual GLsizei width() const = 0;
    virtual GLsizei height() const = 0;

    virtual void load(void*) = 0;
    virtual void bind(GLenum) const = 0;
    virtual void unbind(GLenum) const = 0;
};

};
