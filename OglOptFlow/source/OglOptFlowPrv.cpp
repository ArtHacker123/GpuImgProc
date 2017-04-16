#include "OglOptFlowPrv.h"
#include "OglOptFlow.h"

#include "OglShape.h"
#include "OglFrameBuffer.h"
#include "GeometryCoord.h"

#include "OglIxIyShader.h"
#include "OglOptFlowShader.h"
#include "OglOptFlowNmsShader.h"

using namespace Ogl;

const std::vector<GLushort> OptFlowPrv::index = { 0, 1, 2, 2, 3, 0 };

const std::vector<Ogl::GeometryCoord> OptFlowPrv::coords =
{
    { { -1.0f, +1.0f, +0.0f }, { +0.0f, +1.0f } },
    { { +1.0f, +1.0f, +0.0f }, { +1.0f, +1.0f } },
    { { +1.0f, -1.0f, +0.0f }, { +1.0f, +0.0f } },
    { { -1.0f, -1.0f, +0.0f }, { +0.0f, +0.0f } }
};

const GLenum OptFlowPrv::mBuffs[] =
{
    GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2,
    GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5
};

OptFlowPrv::OptFlowPrv(const cl::Context& ctxt, GLsizei levels)
    :mLevels(levels),
     mCtxtCL(ctxt),
     mUvImg(0),
     mIxImg(levels),
     mIyImg(levels),
     mCompact(ctxt),
     mCount(ctxt, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 1)
{
    mIxIyShader.reset(new Ogl::IxIyShader);
    mOptFlowShader.reset(new Ogl::OptFlowShader);
    mNmsShader.reset(new Ogl::OptFlowNmsShader);

    GLint vcIndex = mIxIyShader->GetAttribLocation("inVc");
    GLint tcIndex = mIxIyShader->GetAttribLocation("inTc");
    mRect.reset(new Ogl::Shape(vcIndex, tcIndex, index, coords));
}

OptFlowPrv::~OptFlowPrv()
{
}

void OptFlowPrv::createImages(GLsizei w, GLsizei h)
{
    if (mIxImg[0].get() == 0 || w != mIxImg[0]->width() || h != mIxImg[0]->height())
    {
        for (size_t i = 0; i < mLevels; i++)
        {
            mIxImg[i].reset(new Ogl::Image<GL_RED>(w, h, GL_R32F, GL_FLOAT));
            mIyImg[i].reset(new Ogl::Image<GL_RED>(w, h, GL_R32F, GL_FLOAT));
            w /= 2; h /= 2;
        }
    }
}

void OptFlowPrv::suppressNonMax(GLfloat minFlowDist)
{
    Ogl::FrameBuffer fb;
    glDrawBuffers(1, mBuffs);
    glViewport(0, 0, mNmsImg->width(), mNmsImg->height());
    fb.bind(GL_DRAW_FRAMEBUFFER);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mNmsImg->texture(), 0);
    mNmsShader->setThreshold(minFlowDist);
    mRect->draw(*mNmsShader, *mUvImg[0]);
    Ogl::FrameBuffer::release(GL_DRAW_FRAMEBUFFER);
}

void OptFlowPrv::computeIxIy(const Ogl::IImage& currImg, const Ogl::IImage& prevImg)
{
    Ogl::FrameBuffer fb;
    for (size_t i = 0; i < mLevels; i++)
    {
        fb.bind(GL_DRAW_FRAMEBUFFER);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mIxImg[i]->texture(), 0);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, mIyImg[i]->texture(), 0);
        glDrawBuffers(2, mBuffs);

        glViewport(0, 0, mIxImg[i]->width(), mIxImg[i]->height());
        mIxIyShader->activate(GL_TEXTURE0);
        prevImg.bind(GL_TEXTURE0);
        currImg.bind(GL_TEXTURE1);
        mRect->draw();
        mIxIyShader->deactivate();

        glDrawBuffers(1, mBuffs);
        Ogl::FrameBuffer::release(GL_DRAW_FRAMEBUFFER);
    }
}

void OptFlowPrv::computeUV(const Ogl::IImage& currImg, const Ogl::IImage& prevImg, GLfloat rvalue)
{
    Ogl::FrameBuffer fb;
    GLsizei w = mIxImg[0]->width();
    GLsizei h = mIxImg[0]->height();

    if (mUvImg.size() == 0 || mUvImg[0]->width() != w || mUvImg[0]->height() != h)
    {
        mUvImg.resize(mLevels+1);
        mNmsImg.reset(new Ogl::Image<GL_RGBA>(w, h, GL_RGBA32F, GL_FLOAT));
        for (size_t i = 0; i < mLevels; i++)
        {
            mUvImg[i].reset(new Ogl::Image<GL_RGBA>(w, h, GL_RGBA32F, GL_FLOAT));
            w /= 2; h /= 2;
        }
        mUvImg[mLevels].reset(new Ogl::Image<GL_RGBA>(w, h, GL_RGBA32F, GL_FLOAT));
    }

    glDrawBuffers(1, mBuffs);
    for (int i = (mLevels-1); i >= 0; i--)
    {
        glViewport(0, 0, mUvImg[i]->width(), mUvImg[i]->height());
        fb.bind(GL_DRAW_FRAMEBUFFER);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mUvImg[i]->texture(), 0);

        mOptFlowShader->SetGain((i == (mLevels-1)) ? 0.0f : 2.0f);
        mOptFlowShader->SetRvalue(rvalue);

        mIxImg[i]->bind(GL_TEXTURE1);
        mIyImg[i]->bind(GL_TEXTURE2);
        currImg.bind(GL_TEXTURE3);
        prevImg.bind(GL_TEXTURE4);
        mRect->draw(*mOptFlowShader, *mUvImg[i+1]);

        Ogl::FrameBuffer::release(GL_DRAW_FRAMEBUFFER);
    }
}

void OptFlowPrv::process(const Ogl::IImage& currImg, const Ogl::IImage& prevImg, GLfloat rvalue, GLfloat minFlowDist)
{
    GLint params[4];
    glGetIntegerv(GL_VIEWPORT, params);
    createImages(currImg.width(), currImg.height());
    computeIxIy(currImg, prevImg);
    computeUV(currImg, prevImg, rvalue);
    suppressNonMax(minFlowDist);
    glViewport(params[0], params[1], params[2], params[3]);
}

bool OptFlowPrv::process(const cl::CommandQueue& queue, Ocl::DataBuffer<Ocl::OptFlowData>& flowData, size_t& outCount, const Ogl::Image<GL_RED>& currImg, const Ogl::Image<GL_RED>& prevImg, GLfloat rvalue, GLfloat minFlowDist)
{
    if (!currImg.isPyramid() || !prevImg.isPyramid())
    {
        return false;
    }
    std::vector<cl::Event> events;
    events.reserve(1024);
    process((const Ogl::IImage&)currImg, (const Ogl::IImage&)prevImg, rvalue, minFlowDist);
    cl::ImageGL imgGL(mCtxtCL, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, mNmsImg->texture());
    std::vector<cl::Memory> gl_objs = { imgGL };
    queue.enqueueAcquireGLObjects(&gl_objs);
    mCompact.process(queue, imgGL, flowData, minFlowDist, mCount, events);
    queue.enqueueReleaseGLObjects(&gl_objs);
    events.back().wait();
    queue.enqueueReadBuffer(mCount.buffer(), CL_TRUE, 0, sizeof(cl_int), &outCount);
    return true;
}
