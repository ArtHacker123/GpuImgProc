#pragma once

#include "OglImage.h"
#include "OglImagePainter.h"

#include "OglBuffer.h"
#include "OglBuffer.h"
#include "OglPainter.h"
#include "OglShape.h"

#include <CL/cl.hpp>

class OglView
{
public:
    OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue);
    ~OglView();

public:
    void draw(uint8_t* pData);
    void resize(GLsizei w, GLsizei h);

private:
    cl::Context& mCtxtCL;
    cl::CommandQueue& mQueueCL;

    Ogl::Image<GL_BGR> mBgrImg;
    Ogl::Image<GL_RED> mGrayImg;

    cl::Kernel mKernel;
    cl::Program mProgram;

    Ogl::ImagePainter< Ogl::RgbaShader, Ogl::Image<GL_BGR> > mRgbaPainter;
    Ogl::ImagePainter< Ogl::LumaShader, Ogl::Image<GL_RED> > mGrayPainter;
};
