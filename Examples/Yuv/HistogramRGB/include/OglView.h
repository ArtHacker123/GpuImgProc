#pragma once

#include "OglImage.h"
#include "OglImagePainter.h"

#include <CL/cl.hpp>
#include "OclHistogramRGB.h"
#include "OglHistogramPainter.h"

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

    Ocl::HistogramRGB mHistogram;
    Ocl::DataBuffer<int> mRgbBins;
    Ogl::HistogramPainter mHistPainter;

    Ogl::Yuv420Image mYuvImg;
    Ogl::Image<GL_RGBA> mRgbaImg;

    Ocl::DataBuffer<int> mRedBuff;
    Ocl::DataBuffer<int> mGreenBuff;
    Ocl::DataBuffer<int> mBlueBuff;

    Ogl::ImagePainter< Ogl::RgbaShader, Ogl::Image<GL_RGBA> > mRgbaPainter;
};
