#include <cuda_runtime.h>
#include <cstdint>

struct Point    { float x, y; };
struct Triangle { Point a, b, c; };

// Fixed-size triangle batches passed by value to CDP child kernels,
// avoiding device-side cudaMalloc.
struct Triangles3  { Triangle t[3]; };
struct Triangles9  { Triangle t[9]; };
struct Triangles27 { Triangle t[27]; };

// ---------- Device utilities ----------

__device__ __forceinline__
bool within_one_pixel(Point a, Point b, Point c, int width, int height,
                      float x_min, float x_max, float y_min, float y_max) {
    float rw = (float)width  / (x_max - x_min);
    float rh = (float)height / (y_max - y_min);
    int pxa = (int)((a.x - x_min) * rw), pya = (int)((a.y - y_min) * rh);
    int pxb = (int)((b.x - x_min) * rw), pyb = (int)((b.y - y_min) * rh);
    int pxc = (int)((c.x - x_min) * rw), pyc = (int)((c.y - y_min) * rh);
    int dist = (pxa-pxb)*(pxa-pxb) + (pya-pyb)*(pya-pyb)
             + (pxb-pxc)*(pxb-pxc) + (pyb-pyc)*(pyb-pyc)
             + (pxa-pxc)*(pxa-pxc) + (pya-pyc)*(pya-pyc);
    return dist < 6;
}

__device__ __forceinline__
Point center_point(Point a, Point b) {
    return {(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f};
}

__device__ __forceinline__
float cross2d(Point a, Point b) { return a.x * b.y - a.y * b.x; }

__device__ __forceinline__
bool is_inside_edge(Point e0, Point e1, Point p) {
    Point ep1 = {p.x - e0.x, p.y - e0.y};
    Point ep2 = {p.x - e1.x, p.y - e1.y};
    return cross2d(ep1, ep2) > 0.0f;
}

__device__ __forceinline__
void draw_point(Point a, uint8_t* pixels, int width, int height,
                float x_min, float x_max, float y_min, float y_max) {
    int px = (int)((a.x - x_min) / (x_max - x_min) * (float)width);
    int py = (int)((a.y - y_min) / (y_max - y_min) * (float)height);
    if (px >= 0 && px < width && py >= 0 && py < height) {
        uint8_t* p = pixels + (py * width + px) * 3;
        p[0] = 255; p[1] = 0; p[2] = 0;
    }
}

__device__ __forceinline__
bool offscreen(Point a, Point b, Point c,
               float x_min, float x_max, float y_min, float y_max) {
    return (a.x <= x_min && b.x <= x_min && c.x <= x_min)
        || (a.x >  x_max && b.x >  x_max && c.x >  x_max)
        || (a.y <= y_min && b.y <= y_min && c.y <= y_min)
        || (a.y >  y_max && b.y >  y_max && c.y >  y_max);
}

// Split one triangle into three by midpoints.
__device__ __forceinline__
void split_triangle(Triangle t, Triangle& ta, Triangle& tb, Triangle& tc) {
    Point cab = center_point(t.a, t.b);
    Point cac = center_point(t.a, t.c);
    Point cbc = center_point(t.b, t.c);
    ta = {t.a, cab, cac};
    tb = {t.b, cbc, cab};
    tc = {t.c, cac, cbc};
}

// ---------- Forward declarations (needed for CDP calls inside render_triangle) ----------

__global__ void fill_triangle_kernel(Point a, Point b, Point c,
                                     uint8_t* pixels, int width, int height,
                                     float x_min, float x_max, float y_min, float y_max);
__global__ void render_kernel_3 (Triangles3  tris, int limit, uint8_t* pixels,
                                  int width, int height,
                                  float x_min, float x_max, float y_min, float y_max);
__global__ void render_kernel_9 (Triangles9  tris, int limit, uint8_t* pixels,
                                  int width, int height,
                                  float x_min, float x_max, float y_min, float y_max);
__global__ void render_kernel_27(Triangles27 tris, int limit, uint8_t* pixels,
                                  int width, int height,
                                  float x_min, float x_max, float y_min, float y_max);

// ---------- Core recursive device function ----------
//
// Mirrors the on-device recursive render_triangle from the AnyDSL source.
// Each acc.exec call inside render_triangle becomes a CDP kernel launch (<<<>>>
// from device code). Child kernels complete before the parent block exits due
// to CDP's implicit block-level synchronization.

__device__ void render_triangle(Point a, Point b, Point c, int limit,
                                uint8_t* pixels, int width, int height,
                                float x_min, float x_max, float y_min, float y_max) {
    if (offscreen(a, b, c, x_min, x_max, y_min, y_max))
        return;

    if (within_one_pixel(a, b, c, width, height, x_min, x_max, y_min, y_max)) {
        draw_point(a, pixels, width, height, x_min, x_max, y_min, y_max);
        return;
    }

    if (limit <= 0) {
        // acc.exec((width, 1, 1), (32, 1, 1)) from fill_triangle — CDP launch.
        fill_triangle_kernel<<<dim3((width + 31) / 32, 1, 1), dim3(32, 1, 1)>>>(
            a, b, c, pixels, width, height, x_min, x_max, y_min, y_max);
        return;
    }

    if (limit >= 3) {
        // acc.exec((27, 1, 1), (27, 1, 1)) — expand three Sierpinski levels at once.
        Triangle ta, tb, tc;
        split_triangle({a, b, c}, ta, tb, tc);

        Triangle taa, tab, tac, tba, tbb, tbc, tca, tcb, tcc;
        split_triangle(ta, taa, tab, tac);
        split_triangle(tb, tba, tbb, tbc);
        split_triangle(tc, tca, tcb, tcc);

        Triangles27 batch;
        split_triangle(taa, batch.t[ 0], batch.t[ 1], batch.t[ 2]);
        split_triangle(tab, batch.t[ 3], batch.t[ 4], batch.t[ 5]);
        split_triangle(tac, batch.t[ 6], batch.t[ 7], batch.t[ 8]);
        split_triangle(tba, batch.t[ 9], batch.t[10], batch.t[11]);
        split_triangle(tbb, batch.t[12], batch.t[13], batch.t[14]);
        split_triangle(tbc, batch.t[15], batch.t[16], batch.t[17]);
        split_triangle(tca, batch.t[18], batch.t[19], batch.t[20]);
        split_triangle(tcb, batch.t[21], batch.t[22], batch.t[23]);
        split_triangle(tcc, batch.t[24], batch.t[25], batch.t[26]);

        render_kernel_27<<<1, 27>>>(batch, limit - 3, pixels, width, height,
                                    x_min, x_max, y_min, y_max);
        return;
    }

    if (limit >= 2) {
        // acc.exec((9, 1, 1), (9, 1, 1)) — expand two levels at once.
        Triangle ta, tb, tc;
        split_triangle({a, b, c}, ta, tb, tc);

        Triangles9 batch;
        split_triangle(ta, batch.t[0], batch.t[1], batch.t[2]);
        split_triangle(tb, batch.t[3], batch.t[4], batch.t[5]);
        split_triangle(tc, batch.t[6], batch.t[7], batch.t[8]);

        render_kernel_9<<<1, 9>>>(batch, limit - 2, pixels, width, height,
                                   x_min, x_max, y_min, y_max);
        return;
    }

    // limit == 1: acc.exec((3, 1, 1), (3, 1, 1)) — one subdivision level.
    Point cab = center_point(a, b);
    Point cac = center_point(a, c);
    Point cbc = center_point(b, c);

    Triangles3 batch;
    batch.t[0] = {a, cab, cac};
    batch.t[1] = {b, cbc, cab};
    batch.t[2] = {c, cac, cbc};

    render_kernel_3<<<1, 3>>>(batch, limit - 1, pixels, width, height,
                               x_min, x_max, y_min, y_max);
}

// ---------- fill_triangle CDP kernel ----------
// Mirrors acc.exec((width, 1, 1), (32, 1, 1)) from AnyDSL fill_triangle,
// launched via CDP from render_triangle when limit <= 0.

__global__ void fill_triangle_kernel(Point a, Point b, Point c,
                                     uint8_t* pixels, int width, int height,
                                     float x_min, float x_max, float y_min, float y_max) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= width) return;
    for (int py = 0; py < height; py++) {
        float x = x_min + (x_max - x_min) * (float)px / (float)width;
        float y = y_min + (y_max - y_min) * (float)py / (float)height;
        Point p = {x, y};
        if (is_inside_edge(b, a, p) && is_inside_edge(c, b, p) && is_inside_edge(a, c, p)) {
            uint8_t* pixel = pixels + (py * width + px) * 3;
            pixel[0] = 0; pixel[1] = 0; pixel[2] = 255;
        }
    }
}

// ---------- render_triangle dispatch kernels ----------
// These are the CDP child kernels. Each thread picks one triangle from the
// batch by threadIdx.x and calls render_triangle as a regular device function.

__global__ void render_kernel_3(Triangles3 tris, int limit, uint8_t* pixels,
                                 int width, int height,
                                 float x_min, float x_max, float y_min, float y_max) {
    int id = threadIdx.x;
    render_triangle(tris.t[id].a, tris.t[id].b, tris.t[id].c, limit,
                    pixels, width, height, x_min, x_max, y_min, y_max);
}

__global__ void render_kernel_9(Triangles9 tris, int limit, uint8_t* pixels,
                                 int width, int height,
                                 float x_min, float x_max, float y_min, float y_max) {
    int id = threadIdx.x;
    render_triangle(tris.t[id].a, tris.t[id].b, tris.t[id].c, limit,
                    pixels, width, height, x_min, x_max, y_min, y_max);
}

__global__ void render_kernel_27(Triangles27 tris, int limit, uint8_t* pixels,
                                  int width, int height,
                                  float x_min, float x_max, float y_min, float y_max) {
    int id = threadIdx.x;
    render_triangle(tris.t[id].a, tris.t[id].b, tris.t[id].c, limit,
                    pixels, width, height, x_min, x_max, y_min, y_max);
}

// ---------- Background fill and entry kernel ----------

__global__ void init_background(uint8_t* pixels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint8_t* p = pixels + (y * width + x) * 3;
    p[0] = 0; p[1] = 255; p[2] = 0;
}

// Mirrors acc.exec((1, 1, 1), (1, 1, 1)) from AnyDSL make_image — the
// single-thread kernel that kicks off the recursive CDP tree.
__global__ void render_entry(Point a, Point b, Point c, int limit,
                              uint8_t* pixels, int width, int height,
                              float x_min, float x_max, float y_min, float y_max) {
    render_triangle(a, b, c, limit, pixels, width, height, x_min, x_max, y_min, y_max);
}

// ---------- Host entry point ----------

extern "C" void make_image(uint8_t* buf, int width, int height,
                            double x_min, double x_max, double y_min, double y_max) {
    size_t size = (size_t)width * height * 3;

    uint8_t* d_pixels;
    cudaMalloc(&d_pixels, size);

    // CDP recursive calls allocate Triangle locals on the thread stack; enlarge
    // the per-thread stack so the worst-case frame (~1.5 KB) fits comfortably.
    cudaDeviceSetLimit(cudaLimitStackSize, 8192);

    // Fill background green — mirrors acc.exec((width, height, 1), (8, 8, 1)).
    dim3 bg_block(8, 8);
    dim3 bg_grid((width + 7) / 8, (height + 7) / 8);
    init_background<<<bg_grid, bg_block>>>(d_pixels, width, height);
    cudaDeviceSynchronize();

    Point a = {0.0f, 0.0f};
    Point b = {-1.0f, 1.414f};
    Point c = { 1.0f, 1.414f};

    render_entry<<<1, 1>>>(a, b, c, 20, d_pixels, width, height,
                            (float)x_min, (float)x_max, (float)y_min, (float)y_max);
    cudaMemcpy(buf, d_pixels, size, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}
