#include "OclScanPrv.h"
#include "OclUtils.h"

#include <sstream>

using namespace Ocl;

ScanPrv::ScanPrv(const cl::Context& ctxt)
    :mWgrpSize(32),
     mDepth(8),
     mBlkSize(1<<8),
     mContext(ctxt),
     mBuffTemp(mContext, CL_MEM_READ_WRITE, mBlkSize)
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

    mProgram = cl::Program(mContext, sSource);
    mProgram.build(options.str().c_str());

    mAdd = cl::Kernel(mProgram, "add");
    mScan = cl::Kernel(mProgram, "scan");
    mGather = cl::Kernel(mProgram, "gather");
}

void ScanPrv::workGroupMultipleAdjust(const cl::CommandQueue& queue)
{
    size_t wgmSize = Ocl::getWorkGroupSizeMultiple(queue, mScan);
    //Intel HD4xxx series GPU's warp_size is not 32
    if (wgmSize != mWgrpSize)
    {
        //re-initialize with wgmSize
        init(wgmSize);
        mWgrpSize = wgmSize;
    }
}

void ScanPrv::adjustEventSize(size_t count, std::vector<cl::Event>& event)
{
    size_t evtSize = 2*(2+(count/(mBlkSize*mBlkSize)));
    if (event.capacity() < (event.size() + evtSize))
    {
        event.reserve(1024);
    }

    if (mWaitList.size() < evtSize)
    {
        mWaitList.resize(evtSize);
        for (size_t i = 0; i < evtSize; i++)
        {
            mWaitList[i].resize(1);
        }
    }
}

void ScanPrv::process(const cl::CommandQueue& queue, Ocl::DataBuffer<int>& buffer,
                        std::vector<cl::Event>& event, std::vector<cl::Event>* pWaitEvent)
{
    int wListCount = 0;
    size_t buffSize = buffer.count();

    adjustEventSize(buffSize, event);
    workGroupMultipleAdjust(queue);

    size_t gSize = (buffSize/mBlkSize);
    gSize += ((buffSize%mBlkSize) == 0) ? 0 : 1;
    gSize *= mBlkSize;

    mScan.setArg(0, buffer.buffer());
    mScan.setArg(1, buffer.buffer());
    mScan.setArg(2, (int)buffSize);
    event.resize(event.size()+1);
    queue.enqueueNDRangeKernel(mScan, cl::NullRange, cl::NDRange(gSize), cl::NDRange(mBlkSize), pWaitEvent, &event.back());

    for (size_t i = mBlkSize; i < buffSize; i += (mBlkSize*mBlkSize))
    {
        mGather.setArg(0, buffer.buffer());
        mGather.setArg(1, mBuffTemp.buffer());
        mGather.setArg(2, (int)(i-1));
        mGather.setArg(3, (int)buffSize);
        mWaitList[wListCount][0] = event.back();
        event.resize(event.size()+1);
        queue.enqueueNDRangeKernel(mGather, cl::NullRange, cl::NDRange(mBlkSize), cl::NDRange(mBlkSize), &mWaitList[wListCount], &event.back());
        wListCount++;

        mAdd.setArg(0, mBuffTemp.buffer());
        mAdd.setArg(1, buffer.buffer());
        mAdd.setArg(2, (int)i);
        mAdd.setArg(3, (int)buffSize);
        mWaitList[wListCount][0] = event.back();
        event.resize(event.size()+1);
        queue.enqueueNDRangeKernel(mAdd, cl::NullRange, cl::NDRange(mBlkSize*mBlkSize), cl::NDRange(mBlkSize), &mWaitList[wListCount], &event.back());
        wListCount++;
    }
}
