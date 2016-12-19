#pragma once

#include "OglImage.h"
#include "GeometryCoord.h"

#include <vector>
#include <memory>
#include <cstdint>

namespace Ogl
{

class Shape;
class FrameBuffer;

class EdgeShader;
class NmesShader;
class GaussShader;
class BinaryShader;

class CannyPrv
{
public:
    CannyPrv();
    ~CannyPrv();

public:
    void process(Ogl::Image<GL_RED>& destImg, const Ogl::Image<GL_RED>& srcImg, GLfloat minThresh, GLfloat maxThresh);
    void process(Ogl::Image<GL_RED>& destImg, const Ogl::Image<GL_LUMINANCE>& srcImg, GLfloat minThresh, GLfloat maxThresh);

protected:
    void createImages(GLsizei w, GLsizei h);
    void doBinary(Ogl::Image<GL_RED>& destImg, const Ogl::IImage& srcImg, GLfloat minThresh, GLfloat maxThresh);
    void doProcess(Ogl::Image<GL_RED>& destImg, const Ogl::IImage& srcImg, GLfloat minThresh, GLfloat maxThresh);
    void doGaussSmooth(Ogl::Image<GL_RED>& destImg, const Ogl::IImage& srcImg);
    void doEdgeAngles(Ogl::Image<GL_RED>& edgeImg, Ogl::Image<GL_RED>& angleImg, const Ogl::IImage& srcImg);
    void doEdgeSuppress(Ogl::Image<GL_RED>& destImg, const Ogl::Image<GL_RED>& edgeImg, const Ogl::Image<GL_RED>& angleImg);

protected:
    std::unique_ptr<Ogl::Shape> mRect;
    std::unique_ptr<Ogl::FrameBuffer> mFb;

    std::unique_ptr<Ogl::NmesShader> mNmesShader;
    std::unique_ptr<Ogl::EdgeShader> mEdgeShader;
    std::unique_ptr<Ogl::GaussShader> mGaussShader;
    std::unique_ptr<Ogl::BinaryShader> mBinaryShader;

    std::unique_ptr< Ogl::Image<GL_RED> > mEdgeImg;
    std::unique_ptr< Ogl::Image<GL_RED> > mAngleImg;
    std::unique_ptr< Ogl::Image<GL_RED> > mSmoothImg;
    std::unique_ptr< Ogl::Image<GL_RED> > mNmesEdgeImg;

    static const GLenum mBuffs[];
    static const std::vector<GLushort> index;
    static const std::vector<Ogl::GeometryCoord> coords;
};

};

