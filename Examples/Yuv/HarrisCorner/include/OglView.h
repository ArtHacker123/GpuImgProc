#pragma once

#include "OglImage.h"
#include "OglImagePainter.h"

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

    Ogl::Yuv420Image mYuvImg;

    Ocl::DataBuffer<Ocl::Pos> mCorners;

    Ocl::HarrisCorner mHarrisCorner;
    Ogl::PointPainter mCornerPainter;

    Ogl::ImagePainter< Ogl::Yuv420Shader, Ogl::Yuv420Image > mYuvPainter;
};
