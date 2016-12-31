#pragma once

#include "OglImage.h"
#include "OglImagePainter.h"

#include "OglBuffer.h"
#include "OglColorShader.h"
#include "OglPainter.h"
#include "OglShape.h"
#include "OglOptFlow.h"
#include "OglOptFlowPainter.h"

#include <CL/cl.hpp>

class GlUvShader;

class OglView
{
public:
    OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue, uint8_t* pData = 0);
    ~OglView();

public:
    void draw(uint8_t* pData);
    void resize(GLsizei w, GLsizei h);

    void thresholdUp();
    void thresholdDown();

protected:
    void swap();

private:
    cl::Context& mCtxtCL;
    cl::CommandQueue& mQueueCL;

    GLfloat mRvalue;
    Ogl::Image<GL_RED>* m_pPrevImg;
    Ogl::Image<GL_RED>* m_pCurrImg;

    Ogl::Image<GL_RED> mImg1;
    Ogl::Image<GL_RED> mImg2;
    Ogl::Image<GL_BGR> mBgrImg;

    Ogl::OptFlow mOptFlow;
    Ocl::DataBuffer<Ocl::OptFlowData> mFlowData;

    Ogl::OptFlowPainter mFlowPainter;
    Ogl::ImagePainter<Ogl::RgbaShader, Ogl::Image<GL_BGR>> mBgrPainter;
};
