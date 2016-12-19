#pragma once

#include <memory>
#include "IOglImage.h"
#include "OglProgram.h"

namespace Ogl
{

class IShader
{
	friend class UseShader;
    friend class UseShaderImage;
public:
	virtual ~IShader() {};
	
	void activate(GLenum tex) { mPgm->use(); ApplyParameters(tex); };
	void deactivate() { mPgm->release(); };
	GLint GetAttribLocation(const GLchar* name) { return mPgm->GetAttribLocation(name); };

protected:
	virtual void ApplyParameters(GLenum tex) = 0;

protected:
	std::unique_ptr<Ogl::Program> mPgm;
};

class UseShader
{
public:
	UseShader(Ogl::IShader& shader);
	~UseShader();

private:
	Ogl::IShader& mShader;
};

class UseShaderImage
{
public:
    UseShaderImage(Ogl::IShader& shader, const Ogl::IImage& img, GLenum tex);
    ~UseShaderImage();

private:
    Ogl::IShader& mShader;
    const Ogl::IImage& mImage;

    GLenum mActTex;
    size_t mTexCount;
};

};
