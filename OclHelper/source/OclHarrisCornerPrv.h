#pragma once

#include "OclCompact.h"
#include "OclDataBuffer.h"

namespace Ocl
{

class HarrisCornerPrv
{
public:
    HarrisCornerPrv(cl::Context& ctxt, cl::CommandQueue& queue);
    ~HarrisCornerPrv();

public:
    size_t process(const cl::Image2D& img, DataBuffer<Ocl::Pos>& corners, float value, size_t& count);
    size_t process(const cl::ImageGL& inImage, DataBuffer<Ocl::Pos>& corners, float value, size_t& count);

private:
    void init();
    void loadGaussCoeffs();
    void checkLocalGroupSizes();
    void createIntImages(const cl::Image& inpImg);
    size_t eigen(const cl::Image& inpImg, cl::Image& outImg);
    size_t gradient(const cl::Image& inpImg, cl::Image& outImg);
    size_t suppress(const cl::Image& inpImg, cl::Image& outImg, float value);

private:
    size_t mWidth;
    size_t mHeight;
    size_t mLocSizeX;
    size_t mLocSizeY;
    cl::Context& mContext;
    cl::CommandQueue& mQueue;

    cl::Program mPgm;

    cl::Kernel mGradKernel;
    cl::Kernel mEigenKernel;
    cl::Kernel mCornerKernel;

    Ocl::DataBuffer<float> mCoeffBuff;
    std::unique_ptr<cl::Image2D> mIxIyImg;
    std::unique_ptr<cl::Image2D> mEigenImg;
    std::unique_ptr<cl::Image2D> mCornerImg;

    Ocl::Compact mCompact;

    static const char sSource[];
};

};

