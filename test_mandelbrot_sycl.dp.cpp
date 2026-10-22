#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include <cstdint>

static inline int round_up(int n, int r) {
    return (n + (r - 1)) / r * r;
}

void mandelbrot_kernel(uint8_t* pixels, int width, int height,
                                  float x_min, float x_range,
                                  float y_min, float y_range,
                                  int max_iter) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int px = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
    int py = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
             item_ct1.get_local_id(1);

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

extern "C" void make_image(uint8_t *buf, int width, int height, double x_min,
                           double x_max, double y_min, double y_max) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    size_t size = (size_t)width * height * 3;

    uint8_t* d_pixels;
    d_pixels = sycl::malloc_device<uint8_t>(size, q_ct1);

    double extent = x_max - x_min;
    int max_iter = (int)(256 + 128 * log2(3.0 / extent));
    if (max_iter < 256)  max_iter = 256;
    if (max_iter > 2048) max_iter = 2048;

    dpct::dim3 block(8, 8, 1);
    dpct::dim3 grid(round_up(width, 8) / 8, round_up(height, 8) / 8, 1);

    q_ct1.submit([&](sycl::handler &cgh) {
        auto x_max_x_min_ct4 = (float)(x_max - x_min);
        auto y_max_y_min_ct6 = (float)(y_max - y_min);

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             mandelbrot_kernel(d_pixels, width, height,
                                               (float)x_min, x_max_x_min_ct4,
                                               (float)y_min, y_max_y_min_ct6,
                                               max_iter);
                         });
    });

    q_ct1.memcpy(buf, d_pixels, size).wait();
    dpct::dpct_free(d_pixels, q_ct1);
}
