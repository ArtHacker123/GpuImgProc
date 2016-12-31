#pragma once

#include "OglImage.h"
#include "OglImagePainter.h"

#include "OglBuffer.h"
#include "OglBuffer.h"
#include "OglPainter.h"
#include "OglShape.h"

#include "OclHarrisCorner.h"
#include "OglPointPainter.h"

class OglView
{
public:
    OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue);
    ~OglView();

public:
    void draw(uint8_t* pData);
    void resize(GLsizei w, GLsizei h);
    void thresholdUp();
    void thresholdDown();

private:
    float mRvalue;
    cl::Context& mCtxtCL;
    cl::CommandQueue& mQueueCL;

    Ogl::Image<GL_BGR> mBgrImg;
    Ogl::Image<GL_RED> mGrayImg;

    Ocl::DataBuffer<Ocl::Pos> mCorners;

    Ocl::HarrisCorner mHarrisCorner;
    Ogl::PointPainter mCornerPainter;

    Ogl::ImagePainter< Ogl::RgbaShader, Ogl::Image<GL_BGR> > mBgrPainter;
};
