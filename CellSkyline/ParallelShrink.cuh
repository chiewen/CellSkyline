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
#include <thrust/sort.h>

#include "Cell.cuh"
#include "Utils.cuh"

void testParallel2();

struct ParallelShrinker {
	template <class T, int D>
	void prepare_cells3(const std::vector<KeyCell<D>>& kc_a, std::vector<Cell<D>>& kc_b, int ce_max, T& t);

	template <class T, int D>
	int shrink_parallel3(const std::vector<KeyCell<D>>& kc_a, std::vector<KeyCell<D>>& kc_b, int ce_max, T& t);

	template <int D>
	std::vector<Cell<D>> process(std::vector<Cell<D>>& cells) const;
};

template <class T, int D>
void ParallelShrinker::prepare_cells3(const std::vector<KeyCell<D>>& kc_a, std::vector<Cell<D>>& kc_b, int ce_max,
                                      T& t) {
	const int Dimension = D;
	Iterator<D - 1> iterator{};
	int cs = kc_a[0].get_last() * 2;
	int ce = ce_max;

	std::map<Iterator<D - 1>, int> m_cs;
	std::map<Iterator<D - 1>, int> m_ce;
	m_cs.insert(std::make_pair(iterator, cs));
	m_ce.insert(std::make_pair(iterator, ce));

	for (int k = 1; k < kc_a.size(); k++) {
		auto& key_cell = kc_a[k];
		auto iter_next = key_cell.get_I().next_layer();

		while (iterator != iter_next) {
			auto fs = m_cs.find(iterator);
			if (fs == m_cs.end()) {
				for (int i = 0; i < Dimension - 1; ++i) {
					auto iter2 = iterator;
					--iter2[i];
					auto f = m_cs.find(iter2);
					if (f != m_cs.end() && f->second > cs) {
						cs = f->second;
					}
				}
				m_cs.insert(std::make_pair(iterator, cs));
			}
			else {
				cs = fs->second;
			}

			ce = ce_max;
			for (int i = 0; i < Dimension - 1; ++i) {
				auto iter2 = iterator;
				--iter2[i];
				auto f = m_ce.find(iter2);
				if (f != m_ce.end() && f->second < ce) {
					ce = f->second;
				}
			}
			for (unsigned short j = cs; j < ce; ++j) {
				Cell<D> cell{iterator[0], iterator[1], j};
				cell.isFilled = t[iterator[0]][iterator[1]][j];

				kc_b.emplace_back(cell);
				if (t[iterator[0]][iterator[1]][j]) {
					auto fe = m_ce.find(iterator);
					if (fe != m_ce.end()) {
						fe->second = j;
					}
					else {
						ce = j + j;
						m_ce.insert(std::make_pair(iterator, ce));
					}
				}
			}
			iterator.advance(ce_max);
		}
		cs = key_cell.get_last() * 2;
		m_cs.insert(std::make_pair(iterator, cs));
	}
	for (unsigned short i = iterator[0]; i < ce_max; ++i) {
		for (unsigned short j = iterator[1]; j < ce_max; ++j) {
			for (unsigned short k = cs; k < ce; ++k) {
				Cell<D> cell{i, j, k};
				cell.isFilled = t[i][j][k];

				kc_b.emplace_back(cell);
				if (t[i][j][k]) {
					ce = k + 1;
				}
			}
		}
	}
}

template <class T, int D>
int ParallelShrinker::shrink_parallel3(const std::vector<KeyCell<D>>& kc_a, std::vector<KeyCell<D>>& kc_b, int ce_max,
                                       T& t) {
	std::vector<Cell<D>> cells;
	prepare_cells3(kc_a, cells, ce_max, t);

	std::vector<KeyCell<3>> key_cells = process(cells);
	return (0);
}

template <int D>
std::vector<Cell<D>> ParallelShrinker::process(std::vector<Cell<D>>& cells) const {
	thrust::device_vector<Cell<D>> d_cells(cells);
	thrust::device_vector<t_pair<D>> d_pair(d_cells.size());
	thrust::device_vector<t_pair<D>> d_result_pair(d_cells.size());
	thrust::transform(d_cells.begin(), d_cells.end(), d_pair.begin(), [] __host__ __device__ (Cell<D>& cell) {
		return thrust::make_pair(cell, cell); });

	
	for (int i = 0; i < D - 1; ++i) {
		if (i > 0) thrust::sort(d_pair.begin(), d_pair.end(), CellPermutation<D>(i));
		thrust::inclusive_scan(d_pair.begin(), d_pair.end(), d_pair.begin(), CellComparer<D>());
	}
	// std::vector<t_pair<D>> h_pair(d_pair.size());
	// thrust::copy(d_pair.begin(), d_pair.end(), h_pair.begin());
	// std::for_each(h_pair.begin(), h_pair.end(), [](t_pair<D> p) {
	// 	std::cout << "{" << p.first << ", " << p.second << "}  ";
	// });

	////////////////////////
	std::vector<t_pair<D>> h_pairs(d_pair.size());
	thrust::copy(d_pair.begin(), d_pair.end(), h_pairs.begin());
	/////////////////

	auto e = thrust::copy_if(d_pair.begin(), d_pair.end(), d_result_pair.begin(), Dominater<D>());

	thrust::device_vector<Cell<D>> d_result(e - d_result_pair.begin());
	std::vector<Cell<D>> results(d_result.size());
	thrust::transform(d_result_pair.begin(), d_result_pair.end(), d_result.begin(), [] __host__ __device__ (const t_pair<D>& p) {
		return p.first;
	});
	thrust::copy(d_result.begin(), d_result.end(), results.begin());
	return results;
}
