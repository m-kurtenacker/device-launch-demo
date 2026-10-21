#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

static inline int round_up(int n, int r) {
    return (n + (r - 1)) / r * r;
}

__global__ void mandelbrot_kernel(uint8_t* pixels, int width, int height,
                                  float x_min, float x_range,
                                  float y_min, float y_range,
                                  int max_iter) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height)
        return;

    float cr = x_min + x_range * (float)px / (float)width;
    float ci = y_min + y_range * (float)py / (float)height;
    float zr = 0.0f, zi = 0.0f;
    int iter = 0;

    while (zr*zr + zi*zi <= 4.0f && iter < max_iter) {
        float tmp = zr*zr - zi*zi + cr;
        zi = 2.0f*zr*zi + ci;
        zr = tmp;
        ++iter;
    }

    uint8_t* pixel = pixels + (py * width + px) * 3;
    if (iter == max_iter) {
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
    } else {
        float t = (float)iter / (float)max_iter;
        pixel[0] = (uint8_t)(9.0f   * (1.0f-t) * t*t*t         * 255.0f);
        pixel[1] = (uint8_t)(15.0f  * (1.0f-t)*(1.0f-t) * t*t  * 255.0f);
        pixel[2] = (uint8_t)(8.5f   * (1.0f-t)*(1.0f-t)*(1.0f-t) * t * 255.0f);
    }
}

extern "C" void make_image(uint8_t* buf, int width, int height,
                           double x_min, double x_max, double y_min, double y_max) {
    size_t size = (size_t)width * height * 3;

    uint8_t* d_pixels;
    cudaMalloc(&d_pixels, size);

    double extent = x_max - x_min;
    int max_iter = (int)(256 + 128 * log2(3.0 / extent));
    if (max_iter < 256)  max_iter = 256;
    if (max_iter > 2048) max_iter = 2048;

    dim3 block(8, 8, 1);
    dim3 grid(round_up(width, 8) / 8, round_up(height, 8) / 8, 1);

    mandelbrot_kernel<<<grid, block>>>(
        d_pixels, width, height,
        (float)x_min, (float)(x_max - x_min),
        (float)y_min, (float)(y_max - y_min),
        max_iter
    );

    cudaMemcpy(buf, d_pixels, size, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}
