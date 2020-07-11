#include "ParallelShrinker.h"

// template<int D>
// struct CellComparer
// {
// 	__host__ __device__
// 	Cell<D> operator()(Cell<D>& ca, Cell<D>& cb) {
// 		for (int i = 2; i < D; ++i) {
// 			if (ca.indices[i] != cb.indices[i]) return cb;
// 		}
// 		if (ca.indices[0] <= cb.indices[0] && ca.indices[1] <= cb.indices[1] && ca.isFilled)
// 			return ca;
// 		return cb;
// 	}
//
// };
//
// int ParallelShrinker::process2(std::vector<Cell<2>>& cells, std::vector<Cell<2>>& key_cells) const {
// 	thrust::device_vector<Cell<2>> d_cells(cells);
// 	thrust::inclusive_scan(d_cells.begin(), d_cells.end(), CellComparer<3>());
// 	key_cells.clear();
// 	key_cells.resize(d_cells.size());
// 	thrust::copy(d_cells.begin(), d_cells.end(), key_cells.begin());
// 	return 0;
// }
//
// int ParallelShrinker::process3(std::vector<Cell<3>>& cells, std::vector<KeyCell<3>>& key_cells) const {
// 	thrust::device_vector<Cell<3>> d_cells(cells);
// 	thrust::inclusive_scan(d_cells.begin(), d_cells.end(), CellComparer<3>());
//
//
// 	return 0;
//
// }

