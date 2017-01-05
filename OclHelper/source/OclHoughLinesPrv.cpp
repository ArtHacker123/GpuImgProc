#include "OclHoughLinesPrv.h"
#include "OclUtils.h"
#include <sstream>

using namespace Ocl;

size_t comp_rho(size_t x, size_t y)
{
    size_t rho = (1 + (size_t)ceil(0.5*sqrt((x*x)+(y*y))));
    rho = (rho/32)+(((rho%32)==0)?0:1);
    return (32*rho);
}

HoughLinesPrv::HoughLinesPrv(cl::Context& ctxt, cl::CommandQueue& queue)
    :mRho(0),
     mContext(ctxt),
     mQueue(queue),
     mCompact(mContext, mQueue)
{
    try
    {
        init();
    }

    catch (cl::Error error)
    {
        fprintf(stderr, "%s", error.what());
        exit(0);
    }
}

HoughLinesPrv::~HoughLinesPrv()
{
}

void HoughLinesPrv::init()
{
    cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
    mPgm = cl::Program(mContext, source);
    mPgm.build();

    mNmsKernel = cl::Kernel(mPgm, "non_max_suppress");
    mHoughKernel = cl::Kernel(mPgm, "hough_line_transform");
}

void HoughLinesPrv::createTempBuffers(const cl::Image& inpImage)
{
    size_t w = 0, h = 0;
    inpImage.getImageInfo<size_t>(CL_IMAGE_WIDTH, &w);
    inpImage.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &h);

    size_t maxEdges = (w*h) >> 2;
    size_t count = (mEdgeData.get() == 0)?0:mEdgeData->count();
    if (count < maxEdges)
    {
        mEdgeData.reset(new Ocl::DataBuffer<Ocl::Pos>(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, maxEdges));
    }

    size_t imgRho = 0;
    mRho = comp_rho(w, h);
    if (mHoughImg.get() != 0)
    {
        mHoughImg->getImageInfo<size_t>(CL_IMAGE_WIDTH, &imgRho);
    }

    if (imgRho != mRho)
    {
        mHoughImg.reset(new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_UNSIGNED_INT32), mRho, 360));
        mHoughNmsImg.reset(new cl::Image2D(mContext, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_UNSIGNED_INT32), mRho, 360));
    }
}

size_t HoughLinesPrv::computeHLT(size_t count)
{
    cl::Event event;
    size_t shMemSize = mRho*sizeof(int);
    mHoughKernel.setArg(0, mEdgeData->buffer());
    mHoughKernel.setArg(1, (int)count);
    mHoughKernel.setArg(2, (int)mRho);
    mHoughKernel.setArg(3, *mHoughImg);
    mHoughKernel.setArg(4, shMemSize, 0);
    mQueue.enqueueNDRangeKernel(mHoughKernel, cl::NullRange, cl::NDRange(360*256), cl::NDRange(256), NULL, &event);
    event.wait();
    return kernelExecTime(mQueue, event);
}

size_t HoughLinesPrv::nonMaxSuppress(size_t threshold)
{
    cl::Event event;
    mNmsKernel.setArg(0, *mHoughImg);
    mNmsKernel.setArg(1, *mHoughNmsImg);
    mNmsKernel.setArg(2, (int)threshold);
    size_t width = (mRho/8)+(((mRho%8)==0)?0:1);
    mQueue.enqueueNDRangeKernel(mNmsKernel, cl::NullRange, cl::NDRange(8*width, 360), cl::NDRange(8, 8), NULL, &event);
    event.wait();
    return kernelExecTime(mQueue, event);
}

size_t HoughLinesPrv::process(const cl::Image2D& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, size_t& houghCount)
{
    size_t edgeCount = 0;
    createTempBuffers(inpImage);
    size_t time = mCompact.process_cartesian(inpImage, *mEdgeData, 1.0, edgeCount);
    time += computeHLT(edgeCount);
    time += nonMaxSuppress(minSize);
    time += mCompact.process(*mHoughNmsImg, hData, (int)minSize, houghCount);
    /*{
        size_t row_pitch = 0;
        size_t slice_pitch = 0;
        cl::size_t<3> img_orig;
        cl::size_t<3> img_region;

        img_region[0] = mRho;
        img_region[1] = 360;
        img_region[2] = 1;

        uint32_t* pdata = (uint32_t *)mQueue.enqueueMapImage(*mHoughNmsImg, CL_TRUE, CL_MAP_READ, img_orig, img_region, &row_pitch, &slice_pitch);
        row_pitch /= sizeof(uint32_t);
        for (size_t y = 0; y < 360; y++)
        {
            uint32_t* pimg_data = (uint32_t*)&pdata[row_pitch*y];
            for (size_t x = 0; x < mRho; x++)
            {
                if (pimg_data[x] > 0)
                {
                    printf("\n(%d, %d) - %d", x, y, pimg_data[x]);
                }
            }
        }
        mQueue.enqueueUnmapMemObject(*mHoughNmsImg, pdata);
    }*/
    return time;
}

size_t HoughLinesPrv::process(const cl::ImageGL& inpImage, size_t minSize, Ocl::DataBuffer<Ocl::HoughData>& hData, size_t& houghCount)
{
    size_t edgeCount = 0;
    createTempBuffers(inpImage);
    std::vector<cl::Memory> gl_objs = { inpImage };
    mQueue.enqueueAcquireGLObjects(&gl_objs);
    size_t time = mCompact.process_cartesian(inpImage, *mEdgeData, 1.0, edgeCount);
    mQueue.enqueueReleaseGLObjects(&gl_objs);
    time += computeHLT(edgeCount);
    time += nonMaxSuppress(minSize);
    time += mCompact.process(*mHoughNmsImg, hData, (int)minSize, houghCount);
    return time;
}
