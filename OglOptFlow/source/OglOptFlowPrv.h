#pragma once

#include "OglImage.h"
#include "OglOptFlow.h"
#include "OptFlowCompact.h"

#include <memory>
#include <vector>

namespace Ogl
{

class Shape;
class FrameBuffer;

class IxIyShader;
class OptFlowShader;
class OptFlowNmsShader;

struct GeometryCoord;

class OptFlowPrv
{
public:
    OptFlowPrv(cl::Context& ctxt, cl::CommandQueue& queue, GLsizei levels);
    ~OptFlowPrv();

public:
    void process(Ocl::DataBuffer<Ocl::OptFlowData>& flowData, size_t& outCount, const Ogl::Image<GL_RED>& currImg, const Ogl::Image<GL_RED>& prevImg, GLfloat rvalue, GLfloat minFlowDist);

protected:
    void suppressNonMax(GLfloat minFlowDist);
    void createImages(GLsizei w, GLsizei h);
    void computeIxIy(const Ogl::IImage& currImg, const Ogl::IImage& prevImg);
    void computeUV(const Ogl::IImage& currImg, const Ogl::IImage& prevImg, GLfloat rvalue);
    void process(const Ogl::IImage& currImg, const Ogl::IImage& prevImg, GLfloat rvalue, GLfloat minFlowDist);

private:
    cl::Context mCtxtCL;
    cl::CommandQueue mQueueCL;

    GLsizei mLevels;
    std::unique_ptr<Ogl::Shape> mRect;

    std::unique_ptr < Ogl::Image<GL_RGBA> > mNmsImg;
    std::vector< std::unique_ptr<Ogl::Image<GL_RGBA> > > mUvImg;

    std::vector< std::unique_ptr< Ogl::Image<GL_RED> > > mIxImg;
    std::vector< std::unique_ptr< Ogl::Image<GL_RED> > > mIyImg;

    std::unique_ptr<Ogl::IxIyShader> mIxIyShader;
    std::unique_ptr<Ogl::OptFlowShader> mOptFlowShader;
    std::unique_ptr<Ogl::OptFlowNmsShader> mNmsShader;

    Ocl::OptFlowCompact mCompact;

    static const GLenum mBuffs[];
    static const std::vector<GLushort> index;
    static const std::vector<Ogl::GeometryCoord> coords;
};

};
