// Runner.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <iterator>

#include "../CellSkyline/DataSet.h"
#include "../CellSkyline/DataSet3.h"
#include "../CellSkyline/shrink_parallel.cuh"
#include <chrono>  

using namespace std::chrono;
using namespace std;

void skyline_2D()
{
	DataSet ds;
	DataSet ds1 (ds);

	std::vector<DataPoint> skyline;
	ds1.skyline_points(ds1.data_points, skyline);

	std::cout << skyline.size() << ":";
	copy(skyline.begin(), skyline.end(), std::ostream_iterator<DataPoint>(std::cout, " "));

	cout << endl << "==========================" << endl;

	skyline = ds.skyline_serial();

	sort(skyline.begin(), skyline.end(), [](const DataPoint& p1, const DataPoint& p2)-> bool
	{
		return p1.x + p1.y < p2.x + p2.y;
	});
	cout << skyline.size() << ":";
	copy(skyline.begin(), skyline.end(), ostream_iterator<DataPoint>(cout, " "));

	cout << endl << "==========================";
}

void print_skyline(vector<DataPointD<3>>& skyline)
{
	sort(skyline.begin(), skyline.end(), [](const DataPoint3& p1, const DataPoint3& p2)-> bool
	{
		return p1[0] < p2[0] || (p1[0] == p2[0] && p1[1] < p2[1]) ||
			(p1[0] == p2[0] && p1[1] == p2[1] && p1[2] < p2[2]);
	});

	std::cout << endl << skyline.size() << ":";
	copy(skyline.begin(), skyline.end(), std::ostream_iterator<DataPoint3>(std::cout, " "));
	
	cout << endl << "==========================" << endl;
}

int main()
{
	// skyline_2D();
	vector<int> times{ 5, 50, 500, 5000, 50000, 500000 };

	for (auto t : times) {
		auto start = system_clock::now();
		DataSet3 ds3(t);
		DataSet3 ds3_1 = ds3;
		vector<DataPoint3> skyline;
		ds3.skyline_points(*ds3.data_points, skyline);
		auto skyline_1 = ds3_1.skyline_serial();
		print_skyline(skyline);
		print_skyline(skyline_1);
		auto end = system_clock::now();
		auto duration = duration_cast<microseconds>(end - start);
		cout << "处理" << t << "个点花费了"
			<< double(duration.count()) * microseconds::period::num / microseconds::period::den
			<< "秒" << endl;
	}	
}

