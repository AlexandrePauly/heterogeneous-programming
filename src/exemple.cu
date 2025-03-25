#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__device__ float gaussian(float x, float sigma) {
    return expf(-(x * x) / (2.0f * sigma * sigma));
}

__global__ void bilateral_filter_cuda(unsigned char *input, unsigned char *output, int width, int height, int filter_radius, float sigma_s, float sigma_r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int index = y * width + x;
    float sum = 0.0f;
    float norm = 0.0f;
    unsigned char center_pixel = input[index];
    
    for (int dy = -filter_radius; dy <= filter_radius; dy++) {
        for (int dx = -filter_radius; dx <= filter_radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                int neighbor_index = ny * width + nx;
                unsigned char neighbor_pixel = input[neighbor_index];
                
                float weight_space = gaussian(sqrtf(dx * dx + dy * dy), sigma_s);
                float weight_intensity = gaussian(abs(center_pixel - neighbor_pixel), sigma_r);
                float weight = weight_space * weight_intensity;
                
                sum += weight * neighbor_pixel;
                norm += weight;
            }
        }
    }
    output[index] = (unsigned char)(sum / norm);
}

void apply_bilateral_filter(unsigned char *h_input, unsigned char *h_output, int width, int height, int filter_radius, float sigma_s, float sigma_r) {
    unsigned char *d_input, *d_output;
    size_t image_size = width * height * sizeof(unsigned char);
    
    cudaMalloc((void **)&d_input, image_size);
    cudaMalloc((void **)&d_output, image_size);
    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    bilateral_filter_cuda<<<gridDim, blockDim>>>(d_input, d_output, width, height, filter_radius, sigma_s, sigma_r);
    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    // Exemple simple : image 512x512 en niveaux de gris
    int width = 512, height = 512;
    int filter_radius = 3;
    float sigma_s = 10.0f, sigma_r = 50.0f;
    
    unsigned char *h_input = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned char *h_output = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    
    // Remplir l'image avec des données (à modifier selon ton code d'entrée)
    for (int i = 0; i < width * height; i++) {
        h_input[i] = rand() % 256;
    }
    
    apply_bilateral_filter(h_input, h_output, width, height, filter_radius, sigma_s, sigma_r);
    
    free(h_input);
    free(h_output);
    return 0;
}
