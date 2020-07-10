#pragma once
#include "DataSet3.h"
#include "DataSet.h"
#include "Cell.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <device_launch_parameters.h>
#include <cstring>
#include <iostream>
#include "DataSet3.h"
#include "DataSet.h"
#include "Cell.h"
#define CHECK(call) \
{ \
const cudaError_t error = call; \
if (error != cudaSuccess) \
{ \
printf("Error: %s:%d, ", __FILE__, __LINE__); \
printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
exit(1); \
} \
}
int shrink_parallel(DataSet3& ds);

int process();
int cudamain();

// template<class T, int D>
// int shrink_parallel3(const std::vector<KeyCell<D>>& kc_a, std::vector<KeyCell<D>>& kc_b, int ce_max, T& t) ;

template <class T, int D>
void prepare_cells3(const std::vector<KeyCell<D>>& kc_a, std::vector<Cell<D>>& kc_b, int ce_max, T& t)
{
	const int Dimension = D;
	Iterator<D - 1> iter{};
	int cs = kc_a[0].get_last() * 2;
	int ce = ce_max;

	std::map<Iterator<D-1>, int> m_cs;
	std::map<Iterator<D-1>, int> m_ce;
	m_cs.insert(std::make_pair(iter, cs));
	m_ce.insert(std::make_pair(iter, ce));
	
	for (int k = 1; k < kc_a.size(); k++)
	{
		auto& key_cell = kc_a[k];
		auto iter_next = key_cell.get_I().next_layer();
		
		while (iter != iter_next)
		{
			auto fs = m_cs.find(iter);
			if (fs == m_cs.end())
			{
				for (int i = 0; i < Dimension - 1; ++i)
				{
					auto iter2 = iter;
					iter2[i]--;
					auto f = m_cs.find(iter2);
					if (f != m_cs.end() && f->second > cs)
					{
						cs = f->second;
					}
				}
				m_cs.insert(std::make_pair(iter, cs));
			}
			else
			{
				cs = fs->second;
			}

			ce = ce_max;
			for (int i = 0; i < Dimension - 1; ++i)
			{
				auto iter2 = iter;
				iter2[i]--;
				auto f = m_ce.find(iter2);
				if (f != m_ce.end() && f->second < ce)
				{
					ce = f->second;
				}
			}
			for (unsigned short j = cs; j < ce; ++j)
			{
				Cell<D> cell{ iter[0], iter[1], j };
				cell.isEmpty = t[iter[0]][iter[1]][j];

				kc_b.emplace_back(cell);
				if (t[iter[0]][iter[1]][j])
				{
					auto fe = m_ce.find(iter);
					if (fe != m_ce.end()) {
						fe->second = j;
					}
					else {
						ce = j + j;
						m_ce.insert(std::make_pair(iter, ce));
					}
				}
			}	
			iter.advance(ce_max);
		}
		cs = key_cell.get_last() * 2;
		m_cs.insert(std::make_pair(iter, cs));
	}
	for (unsigned short i = iter[0]; i < ce_max; ++i)
	{
		for (unsigned short j = iter[1]; j < ce_max; ++j)
		{
			for (unsigned short k = cs; k < ce; ++k)
			{
				Cell<D> cell{i, j, k};
				cell.isEmpty = t[i][j][k];

				kc_b.emplace_back(cell);
				if (t[i][j][k])
				{
					ce = k + 1;
				}
			}
		}
	}

}

template<class T, int D>
int shrink_parallel3(const std::vector<KeyCell<D>>& kc_a, std::vector<KeyCell<D>>& kc_b, int ce_max, T& t) {

	std::vector<Cell<D>> cells;
	prepare_cells3(kc_a, cells, ce_max, t);

	std::cout << "cells:" << cells.size() << std::endl;;

	process();
	cudamain();
	return (0);
}
