#include "pch.h"
#include "../CellSkyline/DataSet.h"
#include "../CellSkyline/KeyCell.h"
#include <iostream>

#include "../CellSkyline/Cell.h"
#include "../CellSkyline/DataSet3.h"
// #include "../CellSkyline/ParallelShrink.cuh"
// #include "../CellSkyline/ParallelShrinker.h"

TEST(SkylineSerial, DataSet) {
	DataSet ds;

	KeyCell<3> kc{ 1, 2, 3 };
	KeyCell<3> kc2{ 1, 2, 3 };
	KeyCell<3> kc3{ 4, 2, 3 };

	std::cout << kc.get_I() << " : " << kc.get_last() << std::endl;

	EXPECT_EQ(ds.data_points.size(), ds.kDataPointNum + 2);
	EXPECT_EQ(kc, kc2);
	EXPECT_NE(kc, kc3);
}
TEST(SkylineSerial, DataPoint) {
	DataPointD<4> dp{ 1, 2, 3, 4 };
	EXPECT_EQ(dp[2], 3);
	EXPECT_EQ(dp[3], 4);
}
TEST(SkylineSerial, DataSet3) {
	KeyCell<5> kc{ 1, 63, 63, 63, 63 };
	auto i = kc.get_I();
	EXPECT_EQ(kc.get_last(), 63);
	i.advance(64);
	auto in = Iterator<4>{ 2, 0, 0, 0 };
	EXPECT_EQ(i, in);
	EXPECT_EQ(i[3], 0);

	Iterator<3> k1{ 2, 3, 5 };
	Iterator<3> k2{ 4, 6, 10 };
	auto k1n = k1.next_layer();
	EXPECT_EQ(k1n[0], k2[0]);
	EXPECT_EQ(k1n[1], k2[1]);
	EXPECT_EQ(k1n[2], k2[2]);
	std::cout << "success" << std::endl;
}

TEST(SkylineSerial, Utils)
{
	Iterator<2> iter{ 2, 3 };
	Iterator<2> iter1{ 4, 8 };
	Iterator<2> iter2{ 1, 9 };
	Iterator<2> iter3{ 1, 3 };
	std::map<Iterator<2>, int> m_cs;
	m_cs.insert(std::make_pair(iter, 18));
	m_cs.insert(std::make_pair(iter1, 19));
	m_cs.insert(std::make_pair(iter2, 28));
	m_cs.insert(std::make_pair(iter3, 17));

	auto aa = m_cs.find(Iterator<2>{1, 9});
	auto bb = m_cs.find(Iterator<2>{1, 10});
	EXPECT_EQ(aa->second, 28);
	EXPECT_EQ(bb, m_cs.end());
}

// TEST(SkylineParallel, Utils)
// {
// 	std::vector<Cell<2>> cells{ {false, 0, 2}, {true, 0, 3}, {false, 0, 4}, {true, 0, 5}, {true, 0, 6}, {true, 0, 7}, {false, 1, 2}, {false, 1, 3}, {false, 1, 4}, {false, 1, 5}, {false, 1, 6}, {false, 1, 7}, {true, 2, 2}, {true, 2, 3}, {true, 3, 2}, {false, 3, 3} };
//
// 	std::vector<Cell<2>> cells2;
// 	// ParallelShrinker ps;
// 	// ps.process2(cells, cells2);
// 	// std::copy(cells2.begin(), cells2.end(), std::ostream_iterator<Cell<2>>(std::cout, " "));
// }