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

template<int D>
struct CellComparer
{
	// __host__ __device__
	// Cell<D> operator()(Cell<D>& ca, Cell<D>& cb) {
	// 	for (int i = 2; i < D; ++i) {
	// 		if (ca.indices[i] != cb.indices[i]) return cb;
	// 	}
	// 	if (ca.indices[0] <= cb.indices[0] && ca.indices[1] <= cb.indices[1] && ca.isFilled)
	// 		return ca;
	// 	return cb;
	// }

	__host__ __device__
		Cell<D> operator()(Cell<D>& ca, Cell<D>& cb) {
		for (int i = 2; i < D; ++i) {
			if (ca.indices[i] != cb.indices[i]) return cb;
		}
		if (ca.indices[0] <= cb.indices[0] && ca.indices[1] <= cb.indices[1] && ca.isFilled)
			return ca;
		if (ca.isFilled && !cb.isFilled)
			return ca;
		return cb;
	}
};

template<int D>
struct Dominater
{
	__host__ __device__
		Cell<D> operator()(Cell<D>& ca, Cell<D>& cb)
	{
		for (int i = 0; i < D; ++i)
		{
			if (cb.indices[i] < ca.indices[i])
				return cb;
		}
		return ca;
	}
};

