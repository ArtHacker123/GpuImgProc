#pragma once

#include "OglImage.h"
#include "OglImagePainter.h"

#include "OglBuffer.h"
#include "OglBuffer.h"
#include "OglPainter.h"
#include "OglShape.h"
#include "OclDataTypes.h"
#include "OclDataBuffer.h"
#include "OglPointPainter.h"
#include "OglColorShader.h"

typedef struct
{
    cl_float rvalue;
    cl_int cornerCount;
    cl_int maxCornerCount;
} CornerParams;

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
    void init();
    void drawCorners();
    void detectCorners(const cl::ImageGL& inpImg);

private:
    cl::Context& mCtxtCL;
    cl::CommandQueue& mQueueCL;

    Ogl::Image<GL_BGR> mBgrImg;
    Ogl::Image<GL_RED> mGrayImg;

    Ogl::Buffer mPointBuff;
    Ogl::Painter<Ogl::ColorShader> mPainter;

    cl::Kernel mKernel;
    cl::Kernel mCrossKernel;
    cl::Program mProgram;

    cl::Image2D mIntImg1;
    cl::Image2D mIntImg2;
    cl_int* m_pIntBuffData;
    cl_int2* m_pCornerData;
    CornerParams* m_pCornerParam;

    Ogl::ImagePainter< Ogl::RgbaShader, Ogl::IImage > mRgbaPainter;

    static const char sSource[];
};
