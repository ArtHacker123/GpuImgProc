#pragma once

#include <boost/compute/core.hpp>
#include <boost/compute/container.hpp>

class InclusiveScan
{
public:
	InclusiveScan(unsigned char depth, boost::compute::context& ctxt, boost::compute::command_queue& queue);
	~InclusiveScan();

public:
	void process(boost::compute::vector<int>& data);

private:
	const int mBlkSize;
	const unsigned char mDepth;
	boost::compute::context& mContext;
	boost::compute::command_queue& mQueue;

	boost::compute::vector<int> mIntData;

	boost::compute::program mProgram;
	
	boost::compute::kernel mScanKernel;
	boost::compute::kernel mAddResKernel;
	boost::compute::kernel mGatherScanKernel;

	static const char mPgmSrc[];
};
