#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "KeyCell.h"
#include "DataSet3.h"
#include "Cell.h"
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <device_launch_parameters.h>

#include <iostream>
#include <thrust/device_vector.h>

void testParallel2();

template<int D>
class Cell
{
public:
	unsigned short int indices[D];

	__host__ __device__
	Cell();
	Cell(std::initializer_list<unsigned short int> il);
	Cell(std::initializer_list<unsigned short int> il, bool isf);
	bool isFilled = true;

	unsigned short int& operator[](int i) { return indices[i]; }
	
	friend std::ostream& operator<< (std::ostream& os, const Cell& p)
	{
		os << "[" << p.isFilled << ": ";
		std::copy(std::begin(p.indices), std::end(p.indices), std::ostream_iterator<unsigned short int>(os, " "));
		os << "]";
		return os;
	}
	
	friend bool operator== (const Cell& kc1, const Cell& kc2)
	{
		return std::equal(std::begin(kc1.indices), std::end(kc1.indices), std::begin(kc2.indices));
	}

	friend bool operator!= (const Cell& kc1, const Cell& kc2)
	{
		return !(kc1 == kc2);
	}
};

template <int D>
Cell<D>::Cell()
{
	for (int i = 0; i < D; ++i)
	{
		indices[i] = -1;
	}
}

template <int D>
Cell<D>::Cell(std::initializer_list<unsigned short int> il) 
{
	int i = 0;
	for (auto elem : il)
	{
		indices[i++] = elem;
	}
}

template <int D>
Cell<D>::Cell(std::initializer_list<unsigned short> il, bool isf) : Cell(il)
{
	this->isFilled = isf;
}


struct ParallelShrinker
{
	template <class T, int D>
	void prepare_cells3(const std::vector<KeyCell<D>>& kc_a, std::vector<Cell<D>>& kc_b, int ce_max, T& t);

	template <class T, int D>
	int shrink_parallel3(const std::vector<KeyCell<D>>& kc_a, std::vector<KeyCell<D>>& kc_b, int ce_max, T& t);

	int process3(std::vector<Cell<3>>& cells, std::vector<KeyCell<3>>& key_cells) const;
	int process2(std::vector<Cell<2>>& cells, std::vector<Cell<2>>& key_cells) const;

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
					iter2[i]--;
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
				iter2[i]--;
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
int ParallelShrinker::shrink_parallel3(const std::vector<KeyCell<D>>& kc_a, std::vector<KeyCell<D>>& kc_b,
                                          int ce_max, T& t) {
	std::vector<Cell<D>> cells;
	prepare_cells3(kc_a, cells, ce_max, t);

	std::cout << "cells:" << cells.size() << std::endl;

	std::vector<KeyCell<3>> key_cells{};
	process3(cells, key_cells);
	return (0);

}
