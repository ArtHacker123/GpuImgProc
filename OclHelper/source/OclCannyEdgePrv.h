#pragma once

#include "OclDataBuffer.h"
#include <memory>

namespace Ocl
{

class CannyEdgePrv
{
public:
    CannyEdgePrv(const cl::Context& ctxt);
    ~CannyEdgePrv();

public:
    size_t process(const cl::CommandQueue& queue, const cl::Image2D& img, cl::Image2D& outImage, float minThresh, float maxThresh);
    size_t process(const cl::CommandQueue& queue, const cl::ImageGL& inImage, cl::ImageGL& outImage, float minThresh, float maxThresh);

private:
    void init();
    void checkLocalGroupSizes();
    void createIntImages(const cl::Image& inpImg);
    void gauss(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, cl::Event& event);
    void gradient(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, cl::Event& waitEvent, cl::Event& event);
    void suppress(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, cl::Event& waitEvent, cl::Event& event);
    void binaryThreshold(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, float minThresh, float maxThresh, cl::Event& waitEvent, cl::Event& event);

private:
    size_t mWidth;
    size_t mHeight;
    size_t mLocSizeX;
    size_t mLocSizeY;
    const cl::Context& mContext;

    cl::Program mPgm;

    cl::Kernel mGradKernel;
    cl::Kernel mGaussKernel;
    cl::Kernel mNmesKernel;
    cl::Kernel mBinThreshKernel;

    std::unique_ptr<cl::Image2D> mGradImg;
    std::unique_ptr<cl::Image2D> mGaussImg;
    std::unique_ptr<cl::Image2D> mNmesImg;

    static const char sSource[];
};

};

