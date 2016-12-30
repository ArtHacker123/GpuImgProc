#pragma once

#include "OglImage.h"
#include "OglImagePainter.h"

#include "OglBuffer.h"
#include "OglColorShader.h"
#include "OclHistogramRGB.h"
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

    Ocl::DataBuffer<int> mRgbBins;
    Ocl::HistogramRGB mHistogram;
    Ogl::HistogramPainter mHistPainter;

    Ocl::DataBuffer<int> mRedBuff;
    Ocl::DataBuffer<int> mGreenBuff;
    Ocl::DataBuffer<int> mBlueBuff;
};
