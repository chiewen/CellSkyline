#include "ParallelShrink.cuh"

void testParallel2() {
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
}
