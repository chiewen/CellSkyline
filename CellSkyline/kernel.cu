#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__device__ int add(int a, int b)
{
	return a + b;
}

__global__ void addKernel(int* c, const int* a, const int* b)
{
	int x = threadIdx.x;
	int y = threadIdx.y;
	int i = y * 3 + x;
	c[i] = add(a[i], b[i]);
}

__global__ void helloFromGPU()
{
	printf("hello from GPU\n");
}

int cudamain()
{
	cudaDeviceProp cdp0, cdp1;
	cudaGetDeviceProperties(&cdp0, 0);

	std::cout << cdp0.name << std::endl;

	const int arraySize = 6;
	const int a[arraySize] = {1, 2, 3, 4, 5, 6};
	const int b[arraySize] = {10, 20, 30, 40, 50, 11};
	int c[arraySize] = {0};

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);

	// helloFromGPU <<<1, 10 >>> ();

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d,%d}\n",
	       c[0], c[1], c[2], c[3], c[4], c[5]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
	int* dev_a = nullptr;
	int* dev_b = nullptr;
	int* dev_c = nullptr;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dim3 block(size / 2, 2);

	// Launch a kernel on the GPU with one thread for each element.
	addKernel<<<1, block>>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
	}

	// cudaDeviceReset();

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
