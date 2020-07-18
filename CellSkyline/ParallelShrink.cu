#include "ParallelShrink.cuh"

std::vector<Cell<3>> ParallelShrinker::shrink_parallel3(DataSet3& data_set3)
{
	std::vector<Cell<3>> cells_l0{{0, 0, 0, false}};
	//
	// std::vector<Cell<3>> cells_l1 = process(cells_l0);
	auto cells_l1 = process(expand_cells3(cells_l0, data_set3.pt1));
	auto cells_l2 = process(expand_cells3(cells_l1, data_set3.pt2));
	auto cells_l3 = process(expand_cells3(cells_l2, data_set3.pt3));
	auto cells_l4 = process(expand_cells3(cells_l3, data_set3.pt4));
	auto cells_l5 = process(expand_cells3(cells_l4, data_set3.pt5));
	auto cells_l6 = process(expand_cells3(cells_l5, data_set3.pt6));
	auto cells_l7 = process(expand_cells3(cells_l6, data_set3.pt7));
	//
	return cells_l7;
}

void testParallel2()
{
	std::vector<Cell<2>> cells{
		{0, 2, false}, {0, 3, true}, {0, 4, false}, {0, 5, true}, {0, 6, true}, {0, 7, true}, {1, 2, false},
		{1, 3, false}, {1, 4, false}, {1, 5, false}, {1, 6, false}, {1, 7, false}, {2, 2, true}, {2, 3, true},
		{3, 2, true}, {3, 3, false}, {4, 0, false}, {4, 1, true}, {4, 2, false}, {4, 3, false}, {5, 0, true},
		{5, 1, false}, {5, 2, false}, {5, 3, false}, {6, 0, false}, {6, 1, false}, {7, 0, true},
		{7, 1, false}
	};

	ParallelShrinker ps;
	std::vector<Cell<2>> cells2 = ps.process(cells);
	std::copy(cells2.begin(), cells2.end(), std::ostream_iterator<Cell<2>>(std::cout, " "));

	std::cout << std::endl;
	std::vector<Cell<3>> cell3{
		{0,2,1,false}, {0,3,4,false},{1,2,2,false}
	};
	std::vector<Cell<3>> cell3a = ps.process(cell3);
	std::copy(cell3a.begin(), cell3a.end(), std::ostream_iterator<Cell<3>>(std::cout, " "));

}
