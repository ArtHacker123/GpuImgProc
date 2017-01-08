#include "OclCompactPrv.h"
#include "OclUtils.h"

#include <sstream>

using namespace Ocl;

CompactPrv::CompactPrv(const cl::Context& ctxt)
    :mWgrpSize(32),
     mScanBlkSize(256),
     mReduceBlkSize(64),
     mContext(ctxt),
     mOutSize(mContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1),
     mScan(ctxt)
{
    try
    {
        init(mWgrpSize);
    }

    catch (cl::Error error)
    {
        fprintf(stderr, "Error: %s", error.what());
        exit(0);
    }
}

CompactPrv::~CompactPrv()
{
}

void CompactPrv::init(size_t warpSize)
{
    std::ostringstream options;
    options << "-DSH_MEM_SIZE_REDUCE=" << 64 << " -DSH_MEM_SIZE=" << 256 << " -DWARP_SIZE=" << warpSize;

    cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
    mProgram = cl::Program(mContext, source);
    mProgram.build(options.str().c_str());

    mReduceFloatX = cl::Kernel(mProgram, "reduce_sum_float_x");
    mCompactFloatX = cl::Kernel(mProgram, "compact_coords_float_x");
    mCompactCartFloatX = cl::Kernel(mProgram, "compact_cartesian_coords_float_x");

    mReduceFloatZ = cl::Kernel(mProgram, "reduce_sum_float_z");
    mCompactOptFlow = cl::Kernel(mProgram, "compact_optflow");

    mReduceIntX = cl::Kernel(mProgram, "reduce_sum_int_x");
    mCompactHoughData = cl::Kernel(mProgram, "compact_hough_data");
}

void CompactPrv::createIntBuffer(size_t buffSize)
{
    size_t memSize = 0;
    size_t intBuffSize = (buffSize / (4 * mReduceBlkSize)) + (((buffSize % (4 * mReduceBlkSize)) == 0) ? 0 : 1);
    if (mBuffReduce.get() != 0)
    {
        memSize = mBuffReduce->count();
    }

    if (memSize < intBuffSize)
    {
        mBuffReduce.reset(new DataBuffer<int>(mContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, intBuffSize));
    }
}

void CompactPrv::workGroupMultipleAdjust(const cl::CommandQueue& queue)
{
    size_t wgmSize = Ocl::getWorkGroupSizeMultiple(queue, mCompactFloatX);
    //Intel HD4xxx series GPU's warp_size is not 32
    if (wgmSize != mWgrpSize)
    {
        //re-initialize with wgmSize
        init(wgmSize);
        mWgrpSize = wgmSize;
    }
}

size_t CompactPrv::process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::Pos>& out, float value, size_t& outCount)
{
    size_t width, height;
    size_t maxOutSize = out.count();
    inpImage.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
    inpImage.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);

    createIntBuffer(width*height);
    workGroupMultipleAdjust(queue);

    mReduceFloatX.setArg(0, inpImage);
    mReduceFloatX.setArg(1, value);
    mReduceFloatX.setArg(2, *mBuffReduce);
    cl::Event event;
    queue.enqueueNDRangeKernel(mReduceFloatX, cl::NullRange, cl::NDRange(width/4, height), cl::NDRange(8, 8), NULL, &event);
    event.wait();
    size_t time = kernelExecTime(queue, event);

    time += mScan.process(queue, *mBuffReduce);

    mCompactFloatX.setArg(0, inpImage);
    mCompactFloatX.setArg(1, value);
    mCompactFloatX.setArg(2, *mBuffReduce);
    mCompactFloatX.setArg(3, out.buffer());
    mCompactFloatX.setArg(4, (int)maxOutSize);
    mCompactFloatX.setArg(5, mOutSize);
    queue.enqueueNDRangeKernel(mCompactFloatX, cl::NullRange, cl::NDRange(width / 2, height), cl::NDRange(16, 8), NULL, &event);
    event.wait();
    time += kernelExecTime(queue, event);

    int* pData = mOutSize.map(queue, CL_TRUE, CL_MAP_READ, 0, 1);
    outCount = *pData;
    mOutSize.unmap(queue, pData);
    return time;
}

size_t CompactPrv::process_cartesian(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::Pos>& coords, float threshold, size_t& count)
{
    size_t width, height;
    size_t maxOutSize = coords.count();
    inpImage.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
    inpImage.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);

    createIntBuffer(width*height);
    workGroupMultipleAdjust(queue);

    mReduceFloatX.setArg(0, inpImage);
    mReduceFloatX.setArg(1, threshold);
    mReduceFloatX.setArg(2, *mBuffReduce);
    cl::Event event;
    queue.enqueueNDRangeKernel(mReduceFloatX, cl::NullRange, cl::NDRange(width/4, height), cl::NDRange(8, 8), NULL, &event);
    event.wait();
    size_t time = kernelExecTime(queue, event);

    time += mScan.process(queue, *mBuffReduce);

    mCompactCartFloatX.setArg(0, inpImage);
    mCompactCartFloatX.setArg(1, threshold);
    mCompactCartFloatX.setArg(2, *mBuffReduce);
    mCompactCartFloatX.setArg(3, coords.buffer());
    mCompactCartFloatX.setArg(4, (int)maxOutSize);
    mCompactCartFloatX.setArg(5, mOutSize);
    queue.enqueueNDRangeKernel(mCompactCartFloatX, cl::NullRange, cl::NDRange(width/2, height), cl::NDRange(16, 8), NULL, &event);
    event.wait();
    time += kernelExecTime(queue, event);

    int* pData = mOutSize.map(queue, CL_TRUE, CL_MAP_READ, 0, 1);
    count = *pData;
    mOutSize.unmap(queue, pData);
    return time;
}

size_t CompactPrv::process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::OptFlowData>& flowData, float threshold, size_t& count)
{
    size_t width, height;
    size_t maxOutSize = flowData.count();
    inpImage.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
    inpImage.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);

    createIntBuffer(width*height);
    workGroupMultipleAdjust(queue);

    mReduceFloatZ.setArg(0, inpImage);
    mReduceFloatZ.setArg(1, threshold);
    mReduceFloatZ.setArg(2, *mBuffReduce);
    cl::Event event;
    queue.enqueueNDRangeKernel(mReduceFloatZ, cl::NullRange, cl::NDRange(width / 4, height), cl::NDRange(8, 8), NULL, &event);
    event.wait();
    size_t time = kernelExecTime(queue, event);

    time += mScan.process(queue, *mBuffReduce);

    mCompactOptFlow.setArg(0, inpImage);
    mCompactOptFlow.setArg(1, threshold);
    mCompactOptFlow.setArg(2, *mBuffReduce);
    mCompactOptFlow.setArg(3, flowData.buffer());
    mCompactOptFlow.setArg(4, (int)maxOutSize);
    mCompactOptFlow.setArg(5, mOutSize);

    queue.enqueueNDRangeKernel(mCompactOptFlow, cl::NullRange, cl::NDRange(width / 2, height), cl::NDRange(16, 8), NULL, &event);
    event.wait();
    time += kernelExecTime(queue, event);

    int* pData = mOutSize.map(queue, CL_TRUE, CL_MAP_READ, 0, 1);
    count = *pData;
    mOutSize.unmap(queue, pData);
    return time;
}

size_t CompactPrv::process(const cl::CommandQueue& queue, const cl::Image& inpImage, Ocl::DataBuffer<Ocl::HoughData>& houghData, size_t threshold, size_t& count)
{
    size_t width, height;
    size_t maxOutSize = houghData.count();
    inpImage.getImageInfo<size_t>(CL_IMAGE_WIDTH, &width);
    inpImage.getImageInfo<size_t>(CL_IMAGE_HEIGHT, &height);

    createIntBuffer(width*height);
    workGroupMultipleAdjust(queue);

    mReduceIntX.setArg(0, inpImage);
    mReduceIntX.setArg(1, (int)threshold);
    mReduceIntX.setArg(2, *mBuffReduce);
    cl::Event event;
    queue.enqueueNDRangeKernel(mReduceIntX, cl::NullRange, cl::NDRange(width/4, height), cl::NDRange(8, 8), NULL, &event);
    event.wait();
    size_t time = kernelExecTime(queue, event);

    time += mScan.process(queue, *mBuffReduce);

    mCompactHoughData.setArg(0, inpImage);
    mCompactHoughData.setArg(1, (int)threshold);
    mCompactHoughData.setArg(2, *mBuffReduce);
    mCompactHoughData.setArg(3, houghData.buffer());
    mCompactHoughData.setArg(4, (int)maxOutSize);
    mCompactHoughData.setArg(5, mOutSize);

    queue.enqueueNDRangeKernel(mCompactHoughData, cl::NullRange, cl::NDRange(width/2, height), cl::NDRange(16, 8), NULL, &event);
    event.wait();
    time += kernelExecTime(queue, event);

    int* pData = mOutSize.map(queue, CL_TRUE, CL_MAP_READ, 0, 1);
    count = *pData;
    mOutSize.unmap(queue, pData);
    return time;
}
