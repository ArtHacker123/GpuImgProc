#pragma once

#include "OglImage.h"
#include "OglImagePainter.h"

#include "OglBuffer.h"
#include "OglBuffer.h"
#include "OglPainter.h"
#include "OglShape.h"

#include "HarrisCorner.h"
#include "OglPointPainter.h"

class GlView
{
public:
    GlView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue);
    ~GlView();

public:
    void draw(uint8_t* pData);
    void resize(GLsizei w, GLsizei h);

private:
    cl::Context& mCtxtCL;
    cl::CommandQueue& mQueueCL;

    Ogl::Image<GL_BGR> mBgrImg;
    Ogl::Image<GL_RED> mGrayImg;

    Ocl::DataBuffer<Ocl::Pos> mCorners;

    Ocl::HarrisCorner mHarrisCorner;
    Ogl::PointPainter mCornerPainter;

    Ogl::ImagePainter< Ogl::RgbaShader, Ogl::Image<GL_BGR> > mBgrPainter;
};
