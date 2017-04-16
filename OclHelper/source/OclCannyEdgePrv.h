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
    void process(const cl::CommandQueue& queue, const cl::Image2D& img, cl::Image2D& outImage, float minThresh, float maxThresh,
                 std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent);
    void process(const cl::CommandQueue& queue, const cl::ImageGL& inImage, cl::ImageGL& outImage, float minThresh, float maxThresh,
                 std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent);

private:
    void init();
    void checkLocalGroupSizes();
    void createIntImages(const cl::Image& inpImg);
    void gradient(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, std::vector<cl::Event>& events);
    void suppress(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, std::vector<cl::Event>& events);
    void binaryThreshold(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, float minThresh, float maxThresh, std::vector<cl::Event>& events);
    void gauss(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent);

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

    std::vector<cl::Event> mWaitEvent[3];

    static const char sSource[];
};

};

