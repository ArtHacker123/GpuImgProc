#pragma once

#include "OglImage.h"
#include "OglImagePainter.h"

#include "OglBuffer.h"
#include "OglBuffer.h"
#include "OglPainter.h"
#include "OglShape.h"
#include "OclCannyEdge.h"

class OglView
{
public:
    OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue);
    ~OglView();

public:
    void draw(uint8_t* pData);
    void resize(GLsizei w, GLsizei h);
    void minThresholdUp();
    void minThresholdDown();
    void maxThresholdUp();
    void maxThresholdDown();

private:
    float mMinThresh;
    float mMaxThresh;
    cl::Context& mCtxtCL;
    cl::CommandQueue& mQueueCL;

    Ocl::CannyEdge mCanny;

    Ogl::Image<GL_BGR> mBgrImg;
    Ogl::Image<GL_RED> mGrayImg;
    Ogl::Image<GL_RED> mBinaryImg;

    Ogl::ImagePainter< Ogl::RgbaShader, Ogl::Image<GL_BGR> > mRgbaPainter;
    Ogl::ImagePainter< Ogl::LumaShader, Ogl::Image<GL_RED> > mGrayPainter;
};
