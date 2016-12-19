#include "InclusiveScan.h"

const char InclusiveScan::mPgmSrc[] = BOOST_STRINGIZE(
__inline__ void up_sweep(local int sh_data[SH_MEM_SIZE])
{
	const int lid = (int)get_local_id(0);
	for (int k = 2; k <= SH_MEM_SIZE; k *= 2)
	{
		int i = mad(k, lid, (k-1));
		if (i < SH_MEM_SIZE)
		{
			sh_data[i] += sh_data[i-(k/2)];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__inline__ void down_sweep(local int sh_data[SH_MEM_SIZE])
{
	const int lid = (int)get_local_id(0);
	if (lid == 0)
	{
		sh_data[SH_MEM_SIZE-1] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int k = SH_MEM_SIZE; k > 1; k /= 2)
	{
		int i = mad(k, lid, (k-1));
		if (i < SH_MEM_SIZE)
		{
			int j = i-(k/2);
			int temp = sh_data[j];
			sh_data[j] = sh_data[i];
			sh_data[i] += temp;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void prefix_sum(__global int* p_data, int count)
{
	local int sh_data[SH_MEM_SIZE];

	int i = (2*get_local_id(0));
	int start = mad(SH_MEM_SIZE, (int)get_group_id(0), i);
	if (start < count)
	{
		sh_data[i] = p_data[start];
		sh_data[i+1] = p_data[start+1];
	}
	else
	{
		sh_data[i] = 0;
		sh_data[i+1] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	up_sweep(sh_data);
	down_sweep(sh_data);

	if (start < count)
	{
		p_data[start] += sh_data[i];
		p_data[start+1] += sh_data[i+1];
	}
}

__kernel void gather_scan(__global read_only int* p_data, int start, int count, __global int* p_out_data)
{
	local int sh_data[SH_MEM_SIZE];

	const int i = (get_local_id(0)*2);
	const int ofs = mad(i, SH_MEM_SIZE, (start-1));
	const int data1 = (ofs < count)?p_data[ofs]:0;
	const int data2 = ((ofs+SH_MEM_SIZE) < count)?p_data[ofs+SH_MEM_SIZE]:0;

	sh_data[i] = data1;
	sh_data[i+1] = data2;
	barrier(CLK_LOCAL_MEM_FENCE);

	up_sweep(sh_data);
	down_sweep(sh_data);

	p_out_data[i] = data1+sh_data[i];
	p_out_data[i+1] = data2+sh_data[i+1];
}

__kernel void add_data(__global int* p_data, int offset, int out_count, __global int* p_in_data)
{
	__local int sh_data;
	if (get_local_id(0) == 0)
	{
		sh_data = p_in_data[get_global_id(0)/SH_MEM_SIZE];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	int i = get_global_id(0) + offset;
	if (i < out_count)
	{
		p_data[i] += sh_data;
	}
});

InclusiveScan::InclusiveScan(unsigned char depth, boost::compute::context& ctxt, boost::compute::command_queue& queue)
	:mBlkSize(1<<(depth-1)),
	 mDepth(depth),
	 mContext(ctxt),
	 mQueue(queue),
	 mIntData(mBlkSize*2, mContext)
{
	std::ostringstream options;
	options << "-DDEPTH=" << depth << " -DSH_MEM_SIZE=" << (1<<depth);
	mProgram = boost::compute::program::create_with_source(mPgmSrc, mContext);
	mProgram.build(options.str());

	mScanKernel = boost::compute::kernel(mProgram, "prefix_sum");
	mAddResKernel = boost::compute::kernel(mProgram, "add_data");
	mGatherScanKernel = boost::compute::kernel(mProgram, "gather_scan");

}

InclusiveScan::~InclusiveScan()
{
}

void InclusiveScan::process(boost::compute::vector<int>& data)
{
	mScanKernel.set_arg(0, data);
	mScanKernel.set_arg(1, (int)data.size());
	boost::compute::event event = mQueue.enqueue_1d_range_kernel(mScanKernel, 0, (data.size()/2), mBlkSize);
	event.wait();

	boost::chrono::microseconds time = event.duration<boost::chrono::microseconds>();
	long long int tot_time = time.count();
	for (size_t i = (mBlkSize*2); i < data.size(); i += (mBlkSize*mBlkSize*4))
	{
		mGatherScanKernel.set_arg(0, data);
		mGatherScanKernel.set_arg(1, (int)i);
		mGatherScanKernel.set_arg(2, (int)data.size());
		mGatherScanKernel.set_arg(3, mIntData);
		boost::compute::event event1 = mQueue.enqueue_1d_range_kernel(mGatherScanKernel, 0, mBlkSize*2, mBlkSize);
		event1.wait();
		time = event1.duration<boost::chrono::microseconds>();
		tot_time += time.count();

		mAddResKernel.set_arg(0, data);
		mAddResKernel.set_arg(1, (int)i);
		mAddResKernel.set_arg(2, (int)data.size());
		mAddResKernel.set_arg(3, mIntData);
		boost::compute::event event2 = mQueue.enqueue_1d_range_kernel(mAddResKernel, 0, (mBlkSize*mBlkSize*4), mBlkSize);
		event2.wait();
		time = event2.duration<boost::chrono::microseconds>();
		tot_time += time.count();
	}
	printf("\nKernel Time: %lld us", tot_time);
}
