#include "OclScanPrv.h"
#include "OclUtils.h"

#include <sstream>

using namespace Ocl;

ScanPrv::ScanPrv(const cl::Context& ctxt)
    :mWgrpSize(32),
     mDepth(8),
     mBlkSize(1<<8),
     mContext(ctxt),
     mIntBuff(mContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, (size_t)(mBlkSize*sizeof(int)))
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

ScanPrv::~ScanPrv()
{
}

void ScanPrv::init(size_t warpSize)
{
    std::ostringstream options;
    options << "-DWARP_SIZE=" << warpSize << " -DSH_MEM_SIZE=" << (1 << mDepth);

    cl::Program::Sources source(1, std::make_pair(sSource, strlen(sSource)));
    mProgram = cl::Program(mContext, source);
    mProgram.build(options.str().c_str());

    mScanKernel = cl::Kernel(mProgram, "prefix_sum");
    mAddResKernel = cl::Kernel(mProgram, "add_data");
    mGatherScanKernel = cl::Kernel(mProgram, "gather_scan");
}

void ScanPrv::workGroupMultipleAdjust(const cl::CommandQueue& queue)
{
    size_t wgmSize = Ocl::getWorkGroupSizeMultiple(queue, mScanKernel);
    //Intel HD4xxx series GPU's warp_size is not 32
    if (wgmSize != mWgrpSize)
    {
        //re-initialize with wgmSize
        init(wgmSize);
        mWgrpSize = wgmSize;
    }
}

size_t ScanPrv::process(const cl::CommandQueue& queue, Ocl::DataBuffer<int>& buffer)
{
    cl::Event event;
    size_t buffSize = buffer.count();

    workGroupMultipleAdjust(queue);

    mScanKernel.setArg(0, buffer.buffer());
    mScanKernel.setArg(1, (int)buffSize);

    size_t gSize = (buffSize/mBlkSize);
    gSize += ((buffSize%mBlkSize) == 0) ? 0 : 1;
    gSize *= mBlkSize;
    queue.enqueueNDRangeKernel(mScanKernel, cl::NullRange, cl::NDRange(gSize/2), cl::NDRange(mBlkSize/2), NULL, &event);
    event.wait();
    size_t time = kernelExecTime(queue, event);

    for (size_t i = mBlkSize; i < buffSize; i += (mBlkSize*mBlkSize))
    {
        mGatherScanKernel.setArg(0, buffer);
        mGatherScanKernel.setArg(1, (int)i);
        mGatherScanKernel.setArg(2, (int)buffSize);
        mGatherScanKernel.setArg(3, mIntBuff);
        queue.enqueueNDRangeKernel(mGatherScanKernel, cl::NullRange, cl::NDRange(mBlkSize/2), cl::NDRange(mBlkSize/2), NULL, &event);
        event.wait();
        time += kernelExecTime(queue, event);

        mAddResKernel.setArg(0, buffer);
        mAddResKernel.setArg(1, (int)i);
        mAddResKernel.setArg(2, (int)buffSize);
        mAddResKernel.setArg(3, mIntBuff);
        queue.enqueueNDRangeKernel(mAddResKernel, cl::NullRange, cl::NDRange(mBlkSize*mBlkSize/4), cl::NDRange(mBlkSize/4), NULL, &event);
        event.wait();
        time += kernelExecTime(queue, event);
    }
    //printf("\nKernel Time: %llf us", ((double)time)/1000.0);
    return time;
}
