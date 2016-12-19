#include <iostream>

#include <boost/compute/core.hpp>
#include <boost/compute/image2d.hpp>
#include <boost/compute/container.hpp>
#include <boost/compute/algorithm/iota.hpp>
#include <boost/compute/algorithm/inclusive_scan.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "InclusiveScan.h"

int main()
{
	const int DEPTH = 8;
	const int BLK_SIZE = 1<<(DEPTH-1);
	// get the default device
	boost::compute::device device = boost::compute::system::default_device();

	std::cout << device.name();

	// create a context for the device
	boost::compute::context context(device);

	// create a command queue
	boost::compute::command_queue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	boost::compute::vector<int> dev_data(640*480, context);
	std::vector<int> host_data(dev_data.size());
	for (size_t i = 0; i < host_data.size(); i++)
	{
		host_data[i] = 1;
	}

	InclusiveScan iscan(DEPTH, context, queue);
	for (size_t i = 0; i < 5; i++)
	{
		boost::compute::copy(host_data.begin(), host_data.end(), dev_data.begin(), queue);
		iscan.process(dev_data);
	}

	std::vector<int> res_data(dev_data.size());
	boost::compute::copy(dev_data.begin(), dev_data.end(), res_data.begin(), queue);

	for (size_t i = 0; i < res_data.size(); i++)
	{
		if (res_data[i] != (int)(i+1))
		{
			printf("\nFailed %d %d", i, res_data[i]);
			break;
		}
	}

	/*boost::compute::vector<int> dev_data1(dev_data.size(), context);
	for (size_t i = 0; i < 10; i++)
	{
		boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
		boost::compute::inclusive_scan(dev_data.begin(), dev_data.end(), dev_data1.begin(), queue);
		queue.finish();
		boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());

		boost::posix_time::time_duration dt = t2 - t1;

		std::cout << "\nExec time " << dt.total_milliseconds();
	}*/

	return 0;
}
