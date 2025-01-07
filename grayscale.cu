#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void rgbToGrayscale(unsigned char *input, unsigned char *output, int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * 3;

    if (x < width && y < height) 
    {
        int r = input[idx];
        int g = input[idx + 1];
        int b = input[idx + 2];
        output[y * width + x] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

extern "C" void cudaGrayscale(unsigned char *input, unsigned char *output, int width, int height)
{
    unsigned char *d_input, *d_output;
    size_t imageSize = width * height * 3 * sizeof(unsigned char);
    size_t graySize = width * height * sizeof(unsigned char);

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, graySize);

    cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

    rgbToGrayscale<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height);

    cudaMemcpy(output, d_output, graySize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
