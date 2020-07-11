#include "ParallelShrink.cuh"

template<int D>
struct CellComparer
{
	__host__ __device__
	Cell<D> operator()(Cell<D>& ca, Cell<D>& cb) {
		for (int i = 2; i < D; ++i) {
			if (ca.indices[i] != cb.indices[i]) return cb;
		}
		if (ca.indices[0] <= cb.indices[0] && ca.indices[1] <= cb.indices[1] && ca.isFilled)
			return ca;
		return cb;
	}

};

int ParallelShrinker::process2(std::vector<Cell<2>>& cells, std::vector<Cell<2>>& key_cells) const {
	thrust::device_vector<Cell<2>> d_cells(cells);
	thrust::inclusive_scan(d_cells.begin(), d_cells.end(), d_cells.begin(), CellComparer<2>());
	key_cells.clear();
	key_cells.resize(d_cells.size());
	thrust::copy(d_cells.begin(), d_cells.end(),  key_cells.begin());
	return 0;
}

int ParallelShrinker::process3(std::vector<Cell<3>>& cells, std::vector<KeyCell<3>>& key_cells) const {
	thrust::device_vector<Cell<3>> d_cells(cells);
	thrust::inclusive_scan(d_cells.begin(), d_cells.end(),  d_cells.begin(),CellComparer<3>());


	return 0;

}

void testParallel2()
{
	std::vector<Cell<2>> cells{ {0, 2, false}, { 0, 3, true}, {0, 4, false}, { 0, 5, true}, { 0, 6, true}, { 0, 7, true}, { 1, 2, false}, { 1, 3, false}, { 1, 4, false}, { 1, 5, false}, { 1, 6, false}, { 1, 7, false}, { 2, 2, true}, { 2, 3, true}, { 3, 2, true}, { 3, 3, false} };

	std::vector<Cell<2>> cells2;
	ParallelShrinker ps;
	ps.process2(cells, cells2);
	std::copy(cells2.begin(), cells2.end(), std::ostream_iterator<Cell<2>>(std::cout, " "));
}
