#pragma once

#include "OglImage.h"
#include "OglImagePainter.h"

#include "OglBuffer.h"
#include "OglColorShader.h"
#include "HistogramRGB.h"
#include "HistogramPainter.h"
#include "OglBuffer.h"
#include "OglPainter.h"
#include "OglShape.h"

#include <CL/cl.hpp>

class GlView
{
public:
	GlView(GLsizei w, GLsizei h, cl::Context& ctxt, cl::CommandQueue& queue);
	~GlView();

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
