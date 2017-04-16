#pragma once

#include "OclCompact.h"
#include "OclDataBuffer.h"

namespace Ocl
{

class HarrisCornerPrv
{
public:
    HarrisCornerPrv(const cl::Context& ctxt);
    ~HarrisCornerPrv();

public:
    void process(const cl::CommandQueue& queue, const cl::Image2D& img, DataBuffer<cl_int2>& corners, float value, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent = 0);
    void process(const cl::CommandQueue& queue, const cl::ImageGL& inImage, DataBuffer<cl_int2>& corners, float value, Ocl::DataBuffer<cl_int>& count, std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent = 0);

private:
    void init();
    void createIntImages(const cl::Image& inpImg);
    void eigen(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, std::vector<cl::Event>& events);
    void suppress(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, float value, std::vector<cl::Event>& events);
    void gradient(const cl::CommandQueue& queue, const cl::Image& inpImg, cl::Image& outImg, std::vector<cl::Event>& events, std::vector<cl::Event>* pWaitEvent);

private:
    size_t mWidth;
    size_t mHeight;
    size_t mLocSizeX;
    size_t mLocSizeY;
    const cl::Context& mContext;

    cl::Program mPgm;

    cl::Kernel mGradKernel;
    cl::Kernel mEigenKernel;
    cl::Kernel mCornerKernel;

    std::unique_ptr<cl::Image2D> mIxIyImg;
    std::unique_ptr<cl::Image2D> mEigenImg;
    std::unique_ptr<cl::Image2D> mCornerImg;

    Ocl::Compact mCompact;

    size_t mWlistCount;
    std::vector< std::vector<cl::Event> > mWaitList;

    static const char sSource[];
};

};

