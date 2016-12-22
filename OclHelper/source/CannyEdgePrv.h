#pragma once

#include "DataBuffer.h"
#include <memory>

namespace Ocl
{

class CannyEdgePrv
{
public:
    CannyEdgePrv(cl::Context& ctxt, cl::CommandQueue& queue);
    ~CannyEdgePrv();

public:
    size_t process(const cl::Image2D& img, cl::Image2D& outImage, float minThresh, float maxThresh);
    size_t process(const cl::ImageGL& inImage, cl::ImageGL& outImage, float minThresh, float maxThresh);

private:
    void init();
    void loadGaussCoeffs();
    void createIntImages(size_t w, size_t h);
    void checkLocalGroupSizes(size_t w, size_t h);
    size_t eigen(const cl::Image& inpImg, cl::Image& outImg);
    size_t gradient(const cl::Image& inpImg, cl::Image& outImg);
    size_t suppress(const cl::Image& inpImg, cl::Image& outImg, float value);

private:
    size_t mSizeX;
    size_t mSizeY;
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

    static const char sSource[];
};

};

