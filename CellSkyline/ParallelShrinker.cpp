#include "ParallelShrinker.h"


int ParallelShrinker::process3(std::vector<Cell<3>>& cells, std::vector<KeyCell<3>>& key_cells) const {

	// const int l = log2(cells.size());
 //
	// const auto dev = 0;
	// cudaSetDevice(dev);
 //
	// const int cell_num = cells.size();
	// const auto n_bytes = l * cell_num * sizeof(Cell<3>);
	// Cell<3>* h_bo = static_cast<Cell<3>*>(malloc(n_bytes));
	// Cell<3>* h_bl = static_cast<Cell<3>*>(malloc(n_bytes));
 //
	// for (int i = 0; i < cells.size(); ++i)
	// {
	// 	h_bo[i] = cells[i];
	// }
 //
	// Cell<3> *d_bo, *d_bl;
	// cudaMalloc(static_cast<Cell<3>**>(&d_bo), n_bytes);
	// cudaMalloc(static_cast<Cell<3>**>(&d_bl), n_bytes);
	// cudaMemcpy(d_bo, h_bo, n_bytes, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_bl, h_bl, n_bytes, cudaMemcpyHostToDevice);
 //
	// dim3 block(32 > cell_num ? cell_num : 32);
	// dim3 grid((cell_num + block.x - 1)/ block.x);
 //
	// for (int i = 1; i <= l; ++i) {
	// 	std::cout << "pow result:" << pow(2, l - i) << std::endl;
 //
	// 	// ProcessBo <<<grid, block>>>(int(pow(2, l - i)), d_bo + (i - 1) * cell_num, d_bo + i * cell_num);
	// 	cudaDeviceSynchronize();
	// }
	// for (int i = l; i >= 0; i--) {
	// 	std::cout << "pow result bl:" << pow(2, l - i) << std::endl;
	// 	// ProcessBl<<<grid, block>>>(pow(2, l - i), d_bl + i * cell_num, d_bl + (i + 1) * cell_num, d_bo + i * cell_num);
	// 	cudaDeviceSynchronize();
	// }
	// // sumArraysOnGPU <<< grid, block >>>(d_Bo, d_Bl);
	// printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);
	// // cudaMemcpy(gpu_ref, d_c, n_bytes, cudaMemcpyDeviceToHost);
	// cudaFree(d_bo);
	// cudaFree(d_bl);
	// free(h_bo);
	// free(h_bl);
	// std::cout << "this is parallel" << std::endl;
 //
 //    cudaDeviceReset();
 //
	// thrust::host_vector<int> H(4);
 //
	// // initialize individual elements
	// H[0] = 14;
	// H[1] = 20;
	// H[2] = 38;
	// H[3] = 46;
 //
	// thrust::device_vector<int> D = H;
	//
	// int sum = thrust::reduce(D.begin(), D.end(), (int)0, thrust::plus<int>());
	//
	// std::cout << "sum:" << sum << std::endl;
	//
	// std::cout << D.size() << ", " << D[0] << std::endl;
	// thrust::copy(D.begin(), D.end(), std::ostream_iterator<int>(std::cout, "\n"));

	return 0;

}

