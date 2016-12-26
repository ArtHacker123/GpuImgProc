#pragma once

#include "OglImage.h"

#include <memory>
#include <vector>

namespace Ogl
{

class Shape;
class FrameBuffer;

class IxIyShader;
class OptFlow3Shader;

struct GeometryCoord;

class OptFlow
{
public:
    OptFlow(GLsizei levels);
    ~OptFlow();

public:
    void process(std::vector< std::unique_ptr< Ogl::Image<GL_RG> > >& uvImg, const Ogl::Image<GL_RED>& currImg, const Ogl::Image<GL_RED>& prevImg, GLfloat rvalue);

protected:
    void createImages(GLsizei w, GLsizei h);
    void computeIxIy(const Ogl::IImage& currImg, const Ogl::IImage& prevImg);
    void computeUV(std::vector< std::unique_ptr< Ogl::Image<GL_RG> > >&uvImg, const Ogl::IImage& currImg, const Ogl::IImage& prevImg, GLfloat rvalue);
    void process(std::vector< std::unique_ptr< Ogl::Image<GL_RG> > >&vuImg, const Ogl::IImage& currImg, const Ogl::IImage& prevImg, GLfloat rvalue);

private:
    GLsizei mLevels;
    std::unique_ptr<Ogl::Shape> mRect;

    std::vector<Ogl::Image<GL_RG>*> m_pUVImg;
    std::unique_ptr<Ogl::Image<GL_RG>> mUVImg;

    std::vector< std::unique_ptr< Ogl::Image<GL_RED> > > mIxImg;
    std::vector< std::unique_ptr< Ogl::Image<GL_RED> > > mIyImg;

    std::unique_ptr<Ogl::IxIyShader> mIxIyShader;
    std::unique_ptr<Ogl::OptFlow3Shader> mOptFlow3Shader;

    static const GLenum mBuffs[];
    static const std::vector<GLushort> index;
    static const std::vector<Ogl::GeometryCoord> coords;
};

};
