#include "DataSet3.h"
#include "shrink_parallel.cuh"

using namespace std;

DataSet3::DataSet3(int num):kDataPointNum(num), data_points(new vector<DataPoint3>()),
pt1(new bool[2][2][2]()),
pt2(new bool[4][4][4]()),
pt3(new bool[8][8][8]()),
pt4(new bool[16][16][16]()),
pt5(new bool[32][32][32]()),
pt6(new bool[64][64][64]()),
pt7(new bool[128][128][128]()),
pp(new int[128][128][128][2]())
{
	init_data_points();
}

void DataSet3::skyline_points(std::vector<DataPointD<3>>& points, std::vector<DataPointD<3>>& result) const
{
	sort(points.begin(), points.end(), [](const DataPoint3& p1, const DataPoint3& p2)-> bool
	{
		return p1[0] + p1[1] + p1[2] < p2[0] + p2[1] + p2[2];
	});

	vector<DataPoint3> skyline;
	for_each(points.begin(), points.end(), [&skyline](DataPoint3& point)-> void
	{
		if (!count_if(skyline.begin(), skyline.end(), [&point](DataPoint3& p_s) -> bool
		{
			return p_s[0] <= point[0] && p_s[1] <= point[1] &&  p_s[2] <= point[2];
		}))
		{
			skyline.push_back(point);
		}
	});
	result.insert(result.end(), skyline.begin(), skyline.end());
}

std::vector<DataPoint3> DataSet3::skyline_parallel()
{
	vector<DataPoint3> skyline;

	sort_data_points(kMaxLayer);

	prepare_cells();
	
	vector<KeyCell<3>> kc_0{ {0,0,0} }, kc_1, kc_2, kc_3, kc_4, kc_5, kc_6, kc_7;

	shrink_parallel3(kc_0, kc_1, 2, pt1);
	// shrink_parallel3(kc_1, kc_2, 4, pt2);
	// shrink_parallel3(kc_2, kc_3, 8, pt3);
	// shrink_parallel3(kc_3, kc_4, 16, pt4);
	// shrink_parallel3(kc_4, kc_5, 32, pt5);
	// shrink_parallel3(kc_5, kc_6, 64, pt6);
	// shrink_parallel3(kc_6, kc_7, 128, pt7);
	//
	// refine(kc_7, skyline, 128, pp);

	return skyline;
}

std::vector<DataPoint3> DataSet3::skyline_serial()
{
	vector<DataPoint3> skyline;

	sort_data_points(kMaxLayer);

	prepare_cells();
	
	vector<KeyCell<3>> kc_0{ {0,0,0} }, kc_1, kc_2, kc_3, kc_4, kc_5, kc_6, kc_7;

	shrink_candidates_serial(kc_0, kc_1, 2, pt1);
	shrink_candidates_serial(kc_1, kc_2, 4, pt2);
	shrink_candidates_serial(kc_2, kc_3, 8, pt3);
	shrink_candidates_serial(kc_3, kc_4, 16, pt4);
	shrink_candidates_serial(kc_4, kc_5, 32, pt5);
	shrink_candidates_serial(kc_5, kc_6, 64, pt6);
	shrink_candidates_serial(kc_6, kc_7, 128, pt7);

	refine(kc_7, skyline, 128, pp);

	return skyline;
}

void DataSet3::init_data_points() const
{
	data_points->reserve(kDataPointNum);
	std::generate_n(back_inserter(*data_points), kDataPointNum, []()-> DataPoint3
	{
		short i1 = (static_cast<short>(rand()) % kWidth);
		short i2 = (static_cast<short>(rand()) % kWidth);
		short i3 = (static_cast<short>(rand()) % kWidth);
		return DataPoint3{ i1, i2, i3 };
	});
	data_points->emplace_back(DataPoint3{0, 0, 32767});
	data_points->emplace_back(DataPoint3{0, 32767, 0});
	data_points->emplace_back(DataPoint3{32767, 0, 0});

	fill_n(pt1[0][0], 1 << 3, 0);
	fill_n(pt2[0][0], 1 << 6, 0);
	fill_n(pt3[0][0], 1 << 9, 0);
	fill_n(pt4[0][0], 1 << 12, 0);
	fill_n(pt5[0][0], 1 << 15, 0);
	fill_n(pt6[0][0], 1 << 18, 0);
	fill_n(pt7[0][0], 1 << 21, 0);

	fill_n(pp[0][0][0], 1 << 22, 0);
}

void DataSet3::prepare_cells()
{
	const auto width = DataSet3::kWidth >> kMaxLayer;

	// layer 6 and points 6
	int i = 0;
	for (auto & p : *data_points)
	{
		pt7[p[0] / width][p[1] / width][p[2] / width] = 1;
		if (pp[p[0]/width][p[1] / width][2] == 0)
		{
			pp[p[0] / width][p[1] / width][p[2] / width][0] = i;
		}
		pp[p[0] / width][p[1] / width][p[2] / width][1] = i + 1;

		i++;
	};

	fill_empty_cells(pt7, pt6, 128);
	fill_empty_cells(pt6, pt5, 64);
	fill_empty_cells(pt5, pt4, 32);
	fill_empty_cells(pt4, pt3, 16);
	fill_empty_cells(pt3, pt2, 8);
	fill_empty_cells(pt2, pt1, 4);
}

void DataSet3::sort_data_points(const int layer)
{
	const auto width = DataSet3::kWidth >> layer;

	sort(data_points->begin(), data_points->end(), [=](const DataPoint3& p1, const DataPoint3& p2)-> bool
	{
		const auto p1_0 = p1[0] / width;
		const auto p1_1 = p1[1] / width;
		const auto p1_2 = p1[2] / width;

		const auto p2_0 = p2[0] / width;
		const auto p2_1 = p2[1] / width;
		const auto p2_2 = p2[2] / width;

		// return p1_0 < p2_0 ||
		// 	(p1_0 == p2_0 && p1_1 < p2_1) ||
		// 	(p1_0 == p2_0 && p1_1 == p2_1 && p1_2 < p2_2) ||
		// 	(p1_0 == p2_0 && p1_1 == p2_1 && p1_2 == p2_2 && p1[0] < p2[0]) ||
		// 	(p1_0 == p2_0 && p1_1 == p2_1 && p1_2 == p2_2 && p1[0] == p2[0] && p1[1] < p2[1]) ||
		// 	(p1_0 == p2_0 && p1_1 == p2_1 && p1_2 == p2_2 && p1[0] == p2[0] && p1[1] == p2[1] && p1[2] < p2[2]);

		if (p1_0 < p2_0) return true;
		if (p1_0 > p2_0) return false;
		if (p1_1 < p2_1) return true;
		if (p1_1 > p2_1) return false;
		if (p1_2 < p2_2) return true;
		if (p1_2 > p2_2) return false;
		if (p1[0] < p2[0]) return true;
		if (p1[0] > p2[0]) return false;
		if (p1[1] < p2[1]) return true;
		if (p1[1] > p2[1]) return false;
		if (p1[2] < p2[2]) return true;
		if (p1[2] > p2[2]) return false;
		return true;
	 });
}

