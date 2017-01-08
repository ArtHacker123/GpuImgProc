#include "OclHistogramPrv.h"

using namespace Ocl;

HistogramPrv::HistogramPrv(const cl::Context& ctxt)
    :mContext(ctxt)
{
    try
    {
        cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
        mPgm = cl::Program(mContext, source);
        mPgm.build();

        // create the kernel
        mAccHist = cl::Kernel(mPgm, "accum_histogram_rgb");
        mTempHistFloat = cl::Kernel(mPgm, "histogram_temp_rgb_float");
        mTempHistUint8 = cl::Kernel(mPgm, "histogram_temp_rgb_uint8");
    }

    catch (cl::Error error)
    {
        fprintf(stderr, "%s", error.what());
        exit(0);
    }

    /*cl_device_id devId = 0;
    queue.getInfo<cl_device_id>(CL_QUEUE_DEVICE, &devId);

    size_t gSize = 0;
    cl::Device device(devId);
    std::string name;
    device.getInfo<std::string>(CL_DEVICE_NAME, &name);
    mAccHist.getWorkGroupInfo<size_t>(device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &gSize);*/
}

HistogramPrv::~HistogramPrv()
{
}

void HistogramPrv::createTempHistBuffer(size_t size)
{
    size_t memSize = (mTempBuff.get() == 0) ? 0 : mTempBuff->count();
    if (memSize != size)
    {
        mTempBuff.reset(new Ocl::DataBuffer<int>(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, size));
    }
}

size_t HistogramPrv::computeTempHist(const cl::CommandQueue& queue, const cl::Image& image, size_t& count)
{
    size_t width, height;
    image.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
    image.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);

    size_t wSizeX = (width / 128);
    if ((width%128)) ++wSizeX;
    size_t wSizeY = (height / 8);

    count = (wSizeX*wSizeY);
    size_t sizeTempHist = 3 * 256 * count;
    createTempHistBuffer(sizeTempHist);

    size_t time = 0;
    cl_image_format format;
    image.getImageInfo<cl_image_format>(CL_IMAGE_FORMAT, &format);

    if (format.image_channel_order == CL_RGBA)
    {
        cl::Event event;
        switch (format.image_channel_data_type)
        {
            case CL_FLOAT:
            case CL_UNORM_INT8:
                // set the kernel arguments
                mTempHistFloat.setArg(0, image);
                mTempHistFloat.setArg(1, *mTempBuff);
                queue.enqueueNDRangeKernel(mTempHistFloat, cl::NullRange, cl::NDRange(32 * wSizeX, height), cl::NDRange(32, 8), NULL, &event);
                event.wait();
                time = (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
                break;

            case CL_UNSIGNED_INT8:
                // set the kernel arguments
                mTempHistUint8.setArg(0, image);
                mTempHistUint8.setArg(1, *mTempBuff);
                queue.enqueueNDRangeKernel(mTempHistUint8, cl::NullRange, cl::NDRange(32 * wSizeX, height), cl::NDRange(32, 8), NULL, &event);
                event.wait();
                time = (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
                break;
        }
    }
    return time;
}

size_t HistogramPrv::accumTempHist(const cl::CommandQueue& queue, size_t count, Ocl::DataBuffer<int>& rgbBins)
{
    mAccHist.setArg(0, mTempBuff->buffer());
    mAccHist.setArg(1, rgbBins.buffer());
    mAccHist.setArg(2, (int)count);

    cl::Event event;
    queue.enqueueNDRangeKernel(mAccHist, cl::NullRange, cl::NDRange(3*256*64), cl::NDRange(64), NULL, &event);
    event.wait();
    size_t time = (event.getProfilingInfo<CL_PROFILING_COMMAND_END>()-event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    return time;
}

size_t HistogramPrv::compute(const cl::CommandQueue& queue, const cl::ImageGL& image, Ocl::DataBuffer<int>& rgbBins)
{
    size_t count = 0;
    std::vector<cl::Memory> gl_objs = { image };
    queue.enqueueAcquireGLObjects(&gl_objs);
    size_t time = computeTempHist(queue, image, count);
    queue.enqueueReleaseGLObjects(&gl_objs);
    time += accumTempHist(queue, count, rgbBins);
    return time;
}

size_t HistogramPrv::compute(const cl::CommandQueue& queue, const cl::Image2D& image, Ocl::DataBuffer<int>& rgbBins)
{
    size_t count = 0;
    size_t time = computeTempHist(queue, image, count);
    time += accumTempHist(queue, count, rgbBins);
    return time;
}
