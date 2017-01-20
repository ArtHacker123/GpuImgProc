#pragma once

#include "OglImage.h"
#include "OglImagePainter.h"

#include "OglBuffer.h"
#include "OglColorShader.h"
#include "OclHistogram.h"
#include "OglHistogramPainter.h"

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
    Ogl::ImagePainter< Ogl::RgbaShader, Ogl::Image<GL_BGR> > mPainter;

    Ocl::DataBuffer<cl_int> mRgbBins;
    Ocl::Histogram mHistogram;
    Ogl::HistogramPainter mHistPainter;

    Ocl::DataBuffer<cl_int> mRedBuff;
    Ocl::DataBuffer<cl_int> mGreenBuff;
    Ocl::DataBuffer<cl_int> mBlueBuff;
};
