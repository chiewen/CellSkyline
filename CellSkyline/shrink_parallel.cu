#include "shrink_parallel.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <device_launch_parameters.h>

void checkResult(float* hostRef, float* gpuRef, const int N) {
	double epsilon = 1.0E-8;
	bool match = 1;
	for (int i = 0; i < N; i++) {
		if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
			match = 0;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match) printf("Arrays match.\n\n");
}

void initialData(float* ip, int size) {
	// generate different seed for random number
	time_t t;
	srand((unsigned)time(&t));
	int iii = 3;
	for (int i = 0; i < size; i++) {
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}
}

void sumArraysOnHost(float* A, float* B, float* C, const int N) {
	for (int idx = 0; idx < N; idx++)
		C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysOnGPU(float* A, float* B, float* C) {
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

int process3(std::vector<Cell<3>> &cells, std::vector<KeyCell<3>> &key_cells) {
	
	int l = log2(cells.size());

	int dev = 0;
	cudaSetDevice(dev);

	int cell_num = cells.size();
	size_t nBytes = l * cell_num * sizeof(Cell<3>);
	Cell<3> *h_Bo, *h_Bl, *hostRef, *gpuRef;
	h_Bo = (Cell<3>*)malloc(nBytes);
	h_Bl = (Cell<3>*)malloc(nBytes);

	for (int i = 0; i < cells.size(); ++i)
	{
		h_Bo[i] = cells[i];
	}

	Cell<3> *d_Bo, *d_Bl, *d_C;
	cudaMalloc((Cell<3>**)&d_Bo, nBytes);
	cudaMalloc((Cell<3>**)&d_Bl, nBytes);
	cudaMemcpy(d_Bo, h_Bo, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Bl, h_Bl, nBytes, cudaMemcpyHostToDevice);
	// invoke kernel at host side
	dim3 block(32);
	dim3 grid(cell_num / block.x);
	sumArraysOnGPU <<< grid, block >>>(d_Bo, d_Bl);
	cudaDeviceSynchronize();
	printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);
	// copy kernel result back to host side
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	// add vector at host side for result checks
	cudaFree(d_Bo);
	cudaFree(d_Bl);
	cudaFree(d_C);
	// free host memory
	free(h_Bo);
	free(h_Bl);
	free(hostRef);
	free(gpuRef);
	std::cout << "this is parallel" << std::endl;
	return 0;
}
// template <class T, int D>
// void prepare_cells3(const std::vector<KeyCell<D>>& kc_a, std::vector<Cell<D>>& kc_b, int ce_max, T& t)
// {
// 	const int Dimension = D;
// 	Iterator<D - 1> iter;
// 	int cs = kc_a[0].get_last() * 2;
// 	int ce = ce_max;
//
// 	std::map<Iterator<D-1>, int> m_cs;
// 	std::map<Iterator<D-1>, int> m_ce;
// 	m_cs.insert(std::make_pair(iter, cs));
// 	m_ce.insert(std::make_pair(iter, ce));
// 	
// 	for (int k = 1; k < kc_a.size(); k++)
// 	{
// 		auto& key_cell = kc_a[k];
// 		auto iter_next = key_cell.get_I().next_layer();
// 		
// 		while (iter != iter_next)
// 		{
// 			auto fs = m_cs.find(iter);
// 			if (fs == m_cs.end())
// 			{
// 				for (int i = 0; i < Dimension - 1; ++i)
// 				{
// 					auto iter2 = iter;
// 					iter2[i]--;
// 					auto f = m_cs.find(iter2);
// 					if (f != m_cs.end() && f->second > cs)
// 					{
// 						cs = f->second;
// 					}
// 				}
// 				m_cs.insert(std::make_pair(iter, cs));
// 			}
// 			else
// 			{
// 				cs = fs->second;
// 			}
//
// 			ce = ce_max;
// 			for (int i = 0; i < Dimension - 1; ++i)
// 			{
// 				auto iter2 = iter;
// 				iter2[i]--;
// 				auto f = m_ce.find(iter2);
// 				if (f != m_ce.end() && f->second < ce)
// 				{
// 					ce = f->second;
// 				}
// 			}
// 			for (unsigned short j = cs; j < ce; ++j)
// 			{
// 				kc_b.emplace_back(Cell<D>{iter[0], iter[1], j});
// 				if (t[iter[0]][iter[1]][j] != 0)
// 				{
// 					auto fe = m_ce.find(iter);
// 					if (fe != m_ce.end()) {
// 						fe->second = j;
// 					}
// 					else {
// 						ce = j;
// 						m_ce.insert(std::make_pair(iter, ce));
// 					}
// 				}
// 			}	
// 			iter.advance(ce_max);
// 		}
// 		cs = key_cell.get_last() * 2;
// 		m_cs.insert(std::make_pair(iter, cs));
// 	}
// 	for (unsigned short i = iter[0]; i < ce_max; ++i)
// 	{
// 		for (unsigned short j = iter[1]; j < ce_max; ++j)
// 		{
// 			for (unsigned short k = cs; k < ce; ++k)
// 			{
// 				if (t[i][j][k] != 0)
// 				{
// 					kc_b.push_back(Cell<3>{ i, j, k });
// 					ce = k;
// 					break;
// 				}
// 			}
// 		}
// 	}
//
// }
//
// template<class T, int D>
// int shrink_parallel3(const std::vector<KeyCell<D>>& kc_a, std::vector<Cell<D>>& kc_b, int ce_max, T& t) {
//
// 	std::vector<Cell<D>> cells;
// 	prepare_cells3(kc_a, cells, ce_max, t);
//
// 	int dev = 0;
// 	cudaSetDevice(dev);
// 	// set up data size of vectors
// 	int nElem = 32;
// 	size_t nBytes = nElem * sizeof(float);
// 	float *h_A, *h_B, *hostRef, *gpuRef;
// 	h_A = (float*)malloc(nBytes);
// 	h_B = (float*)malloc(nBytes);
// 	hostRef = (float*)malloc(nBytes);
// 	gpuRef = (float*)malloc(nBytes);
//
// 	// initialize data at host side
// 	initialData(h_A, nElem);
// 	initialData(h_B, nElem);
// 	memset(hostRef, 0, nBytes);
// 	memset(gpuRef, 0, nBytes);
// 	// malloc device global memory
// 	float *d_A, *d_B, *d_C;
// 	cudaMalloc((float**)&d_A, nBytes);
// 	cudaMalloc((float**)&d_B, nBytes);
// 	cudaMalloc((float**)&d_C, nBytes);
// 	// transfer data from host to device
// 	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
// 	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
// 	// invoke kernel at host side
// 	dim3 block(nElem);
// 	dim3 grid(nElem / block.x);
// 	sumArraysOnGPU << < grid, block >> >(d_A, d_B, d_C);
// 	cudaDeviceSynchronize();
// 	printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);
// 	// copy kernel result back to host side
// 	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
// 	// add vector at host side for result checks
// 	sumArraysOnHost(h_A, h_B, hostRef, nElem);
// 	// check device results
// 	checkResult(hostRef, gpuRef, nElem);
// 	// free device global memory
// 	cudaFree(d_A);
// 	cudaFree(d_B);
// 	cudaFree(d_C);
// 	// free host memory
// 	free(h_A);
// 	free(h_B);
// 	free(hostRef);
// 	free(gpuRef);
// 	std::cout << "this is parallel" << std::endl;
// 	return (0);
// }
int shrink_parallel(DataSet3& ds) {
	printf("%s Starting...\n", "hello");
	// set up device
	int dev = 0;
	cudaSetDevice(dev);
	// set up data size of vectors
	int nElem = 32;
	printf("Vector size %d\n", nElem);
	// malloc host memory
	size_t nBytes = nElem * sizeof(float);
	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float*)malloc(nBytes);
	h_B = (float*)malloc(nBytes);
	hostRef = (float*)malloc(nBytes);
	gpuRef = (float*)malloc(nBytes);

	// initialize data at host side
	initialData(h_A, nElem);
	initialData(h_B, nElem);
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);
	// malloc device global memory
	float *d_A, *d_B, *d_C;
	cudaMalloc((float**)&d_A, nBytes);
	cudaMalloc((float**)&d_B, nBytes);
	cudaMalloc((float**)&d_C, nBytes);
	// transfer data from host to device
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
	// invoke kernel at host side
	dim3 block(nElem);
	dim3 grid(nElem / block.x);
	sumArraysOnGPU << < grid, block >> >(d_A, d_B, d_C);
	cudaDeviceSynchronize();
	printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);
	// copy kernel result back to host side
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	// add vector at host side for result checks
	sumArraysOnHost(h_A, h_B, hostRef, nElem);
	// check device results
	checkResult(hostRef, gpuRef, nElem);
	// free device global memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	// free host memory
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);
	std::cout << "this is parallel" << std::endl;
	return (0);
}
