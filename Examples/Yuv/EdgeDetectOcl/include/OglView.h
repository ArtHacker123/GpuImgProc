#pragma once

#include "OglImage.h"
#include "OglImagePainter.h"

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
    cl::Context& mCtxtCL;
    cl::CommandQueue& mQueueCL;

    Ocl::CannyEdge mCanny;

    Ogl::Yuv420Image mYuvImg;
    Ogl::Image<GL_RED> mEdgeImg;

    float mMinThresh;
    float mMaxThresh;

    Ogl::ImagePainter< Ogl::Yuv420Shader, Ogl::Yuv420Image > mYuvPainter;
    Ogl::ImagePainter< Ogl::LumaShader, Ogl::Image<GL_RED> > mLumaPainter;
};
