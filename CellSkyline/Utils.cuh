#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "KeyCell.h"
#include "DataSet3.h"
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <device_launch_parameters.h>

#include <iostream>
#include <thrust/device_vector.h>

#include "Cell.cuh"


template <int D>
using t_pair = thrust::pair<Cell<D>, Cell<D>>;

//
// template <int D>
// struct CellComparerOriginal {
// 	__host__ __device__
//
// 	Cell<D> operator()(Cell<D>& ca, Cell<D>& cb) {
// 		for (int i = 2; i < D; ++i) {
// 			if (ca.indices[i] != cb.indices[i]) return cb;
// 		}
// 		if (ca.indices[0] <= cb.indices[0] && ca.indices[1] <= cb.indices[1] && ca.isFilled)
// 			return ca;
// 		return cb;
// 	}
// };

// template<int D>
// struct CellPairMaker {
// 	__host__ __device__
// 	t_pair<D> operator()(Cell<D> &cell) {
// 		return thrust::make_pair(cell, cell);
// 	}
// };
//
// template <int D>
// struct CandidateCellComparer {
// 	__host__ __device__
//
// 	Cell<D> operator()(Cell<D>& ca, Cell<D>& cb) {
// 		for (int i = 2; i < D; ++i) {
// 			if (ca.indices[i] != cb.indices[i]) return cb;
// 		}
// 		if (ca.indices[0] < cb.indices[0] && ca.indices[1] < cb.indices[1] && ca.isFilled) {
// 			cb.isDominated = true;
// 			return ca;
// 		}
// 		if (ca.indices[0] <= cb.indices[0] && ca.indices[1] <= cb.indices[1] && ca.isFilled)
// 			return ca;
// 		return cb;
// 	}
// };

template <int D>
struct CellComparer {
	__host__ __device__

	t_pair<D> operator()(t_pair<D>& ca, t_pair<D>& cb) {
		for (int i = 2; i < D; ++i) {
			if (ca.second.indices[i] != cb.second.indices[i]) return cb;
		}
		if (ca.second.indices[0] <= cb.second.indices[0] && ca.second.indices[1] <= cb.second.indices[1] && ca.second.isFilled) {
			cb.second = ca.second;
			return cb;
		}
		if (ca.second.isFilled && !cb.second.isFilled) {
			cb.second = ca.second;
			return cb;
		}
		return cb;
	}
};

template <int D>
struct CellPermutation {
	int _p;
	CellPermutation<D>(int p) : _p(p) {}
	__host__ __device__

	bool operator()(t_pair<D>& l, t_pair<D>& r) {
		for (int i = _p; i < D; ++i) {
			if (l.first.indices[i] < r.first.indices[i]) return true;
			if (l.first.indices[i] > r.first.indices[i]) return false;
		}
		for (int i = 0; i < _p; ++i) {
			if (l.first.indices[i] < r.first.indices[i]) return true;
			if (l.first.indices[i] > r.first.indices[i]) return false;
		}
		return false;
	}
};

template <int D>
struct Dominater {
	__host__ __device__

	bool operator()(const t_pair<D>& ca) {
		for (int i = 0; i < D; ++i) {
			if (ca.first.indices[i] == ca.second.indices[i]) {
				return true;
			}
		}
		return false;
	}
};
