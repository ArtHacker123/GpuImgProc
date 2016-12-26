#include "OglOptFlow.h"

#include "OglShape.h"
#include "OglFrameBuffer.h"
#include "GeometryCoord.h"

#include "OglIxIyShader.h"
#include "OglOptFlow3Shader.h"

using namespace Ogl;

const std::vector<GLushort> OptFlow::index = { 0, 1, 2, 2, 3, 0 };

const std::vector<Ogl::GeometryCoord> OptFlow::coords =
{
    { { -1.0f, +1.0f, +0.0f }, { +0.0f, +1.0f } },
    { { +1.0f, +1.0f, +0.0f }, { +1.0f, +1.0f } },
    { { +1.0f, -1.0f, +0.0f }, { +1.0f, +0.0f } },
    { { -1.0f, -1.0f, +0.0f }, { +0.0f, +0.0f } }
};

const GLenum OptFlow::mBuffs[] =
{
    GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2,
    GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5
};

OptFlow::OptFlow(GLsizei levels)
    :mLevels(levels),
     m_pUVImg(levels+1),
     mIxImg(levels),
     mIyImg(levels)
{
    mIxIyShader.reset(new Ogl::IxIyShader);
    mOptFlow3Shader.reset(new Ogl::OptFlow3Shader);

    GLint vcIndex = mIxIyShader->GetAttribLocation("inVc");
    GLint tcIndex = mIxIyShader->GetAttribLocation("inTc");
    mRect.reset(new Ogl::Shape(vcIndex, tcIndex, index, coords));
}

OptFlow::~OptFlow()
{
}

void OptFlow::createImages(GLsizei w, GLsizei h)
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

void OptFlow::computeIxIy(const Ogl::IImage& currImg, const Ogl::IImage& prevImg)
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

void OptFlow::computeUV(std::vector< std::unique_ptr<Ogl::Image<GL_RG>> >& uvImg, const Ogl::IImage& currImg, const Ogl::IImage& prevImg, GLfloat rvalue)
{
    Ogl::FrameBuffer fb;
    GLsizei w = mIxImg[0]->width();
    GLsizei h = mIxImg[0]->height();

    if (uvImg.size() == 0 || uvImg[0]->width() != w || uvImg[0]->height() != h)
    {
        uvImg.resize(mLevels);
        for (size_t i = 0; i < mLevels; i++)
        {
            uvImg[i].reset(new Ogl::Image<GL_RG>(w, h, GL_RGBA32F, GL_FLOAT));
            m_pUVImg[i] = uvImg[i].get();

            w /= 2;
            h /= 2;
        }
        mUVImg.reset(new Ogl::Image<GL_RG>(w, h, GL_RGBA32F, GL_FLOAT));
        m_pUVImg[mLevels] = mUVImg.get();
    }

    glDrawBuffers(1, mBuffs);
    for (int i = (mLevels-1); i >= 0; i--)
    {
        glViewport(0, 0, m_pUVImg[i]->width(), m_pUVImg[i]->height());
        fb.bind(GL_DRAW_FRAMEBUFFER);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_pUVImg[i]->texture(), 0);

        mOptFlow3Shader->SetGain((i == (mLevels-1)) ? 0.0f : 2.0f);
        mOptFlow3Shader->SetRvalue(rvalue);

        mIxImg[i]->bind(GL_TEXTURE1);
        mIyImg[i]->bind(GL_TEXTURE2);
        currImg.bind(GL_TEXTURE3);
        prevImg.bind(GL_TEXTURE4);
        mRect->draw(*mOptFlow3Shader, *m_pUVImg[i+1]);

        Ogl::FrameBuffer::release(GL_DRAW_FRAMEBUFFER);
    }
}

void OptFlow::process(std::vector< std::unique_ptr<Ogl::Image<GL_RG>> >& uvImg, const Ogl::IImage& currImg, const Ogl::IImage& prevImg, GLfloat rvalue)
{
    GLint params[4];
    glGetIntegerv(GL_VIEWPORT, params);
    createImages(currImg.width(), currImg.height());
    computeIxIy(currImg, prevImg);
    computeUV(uvImg, currImg, prevImg, rvalue);
    glViewport(params[0], params[1], params[2], params[3]);
}

void OptFlow::process(std::vector< std::unique_ptr<Ogl::Image<GL_RG>> >& uvImg, const Ogl::Image<GL_RED>& currImg, const Ogl::Image<GL_RED>& prevImg, GLfloat rvalue)
{
    process(uvImg, (const Ogl::IImage&)currImg, (const Ogl::IImage&)prevImg, rvalue);
}
