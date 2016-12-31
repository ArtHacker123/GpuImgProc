#pragma once

#include "OglImage.h"
#include "OglImagePainter.h"

#include "OglOptFlow.h"
#include "OglOptFlowPainter.h"

class OglView
{
public:
    OglView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue);
    ~OglView();

public:
    void draw(uint8_t* pData);
    void resize(GLsizei w, GLsizei h);

private:
    void swap();

private:
    cl::Context& mCtxtCL;
    cl::CommandQueue& mQueueCL;

    GLfloat mRvalue;
    Ogl::Yuv420Image* m_pPrevImg;
    Ogl::Yuv420Image* m_pCurrImg;

    Ogl::Yuv420Image mImg1;
    Ogl::Yuv420Image mImg2;

    Ogl::OptFlow mOptFlow;
    Ocl::DataBuffer<Ocl::OptFlowData> mFlowData;

    Ogl::OptFlowPainter mFlowPainter;

    Ogl::ImagePainter< Ogl::Yuv420Shader, Ogl::Yuv420Image > mYuvPainter;
};
