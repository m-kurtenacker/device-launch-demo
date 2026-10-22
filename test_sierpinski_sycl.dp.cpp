#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <cstdint>

struct Point    { float x, y; };
struct Triangle { Point a, b, c; };

// ---------- Device utilities ----------

__dpct_inline__ bool within_one_pixel(Point a, Point b, Point c, int width,
                                      int height, float x_min, float x_max,
                                      float y_min, float y_max) {
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

__dpct_inline__ Point center_point(Point a, Point b) {
    return {(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f};
}

__dpct_inline__ float cross2d(Point a, Point b) {
    return a.x * b.y - a.y * b.x;
}

__dpct_inline__ bool is_inside_edge(Point e0, Point e1, Point p) {
    Point ep1 = {p.x - e0.x, p.y - e0.y};
    Point ep2 = {p.x - e1.x, p.y - e1.y};
    return cross2d(ep1, ep2) > 0.0f;
}

__dpct_inline__ void draw_point(Point a, uint8_t *pixels, int width, int height,
                                float x_min, float x_max, float y_min,
                                float y_max) {
    int px = (int)((a.x - x_min) / (x_max - x_min) * (float)width);
    int py = (int)((a.y - y_min) / (y_max - y_min) * (float)height);
    if (px >= 0 && px < width && py >= 0 && py < height) {
        uint8_t* p = pixels + (py * width + px) * 3;
        p[0] = 255; p[1] = 0; p[2] = 0;
    }
}

__dpct_inline__ bool offscreen(Point a, Point b, Point c, float x_min,
                               float x_max, float y_min, float y_max) {
    return (a.x <= x_min && b.x <= x_min && c.x <= x_min)
        || (a.x >  x_max && b.x >  x_max && c.x >  x_max)
        || (a.y <= y_min && b.y <= y_min && c.y <= y_min)
        || (a.y >  y_max && b.y >  y_max && c.y >  y_max);
}

// Split one triangle into three by midpoints.
__dpct_inline__ void split_triangle(Triangle t, Triangle &ta, Triangle &tb,
                                    Triangle &tc) {
    Point cab = center_point(t.a, t.b);
    Point cac = center_point(t.a, t.c);
    Point cbc = center_point(t.b, t.c);
    ta = {t.a, cab, cac};
    tb = {t.b, cbc, cab};
    tc = {t.c, cac, cbc};
}

// ---------- One recursion level of render_triangle ----------
//
// The CUDA version recurses on the device through CDP kernel launches. SYCL
// has neither device-side kernel launches nor recursion in device code, so the
// recursion is unrolled into waves driven by the host. Every triangle in a wave
// has the same limit, so the work-item for triangle t does what one call of
// render_triangle(t, limit) did, but instead of launching a child kernel it
// appends the children (or, for limit <= 0, the triangle to fill) to `out`.
// `depth` is the number of subdivision levels expanded at once (3, 2 or 1,
// matching the 27/9/3-thread child launches), or 0 for the fill case.

__dpct_inline__ void expand_triangle(Triangle t, int depth, Triangle *out,
                                     uint32_t *out_count, uint8_t *pixels,
                                     int width, int height, float x_min,
                                     float x_max, float y_min, float y_max) {
    if (offscreen(t.a, t.b, t.c, x_min, x_max, y_min, y_max))
        return;

    if (within_one_pixel(t.a, t.b, t.c, width, height, x_min, x_max, y_min, y_max)) {
        draw_point(t.a, pixels, width, height, x_min, x_max, y_min, y_max);
        return;
    }

    int num_children = depth == 3 ? 27 : depth == 2 ? 9 : depth == 1 ? 3 : 1;
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        counter(*out_count);
    uint32_t base = counter.fetch_add((uint32_t)num_children);

    // Child k is found by following the base-3 digits of k down `depth`
    // levels of split_triangle.
    for (int k = 0; k < num_children; k++) {
        Triangle s = t;
        int code = k;
        for (int l = 0; l < depth; l++) {
            Triangle ta, tb, tc;
            split_triangle(s, ta, tb, tc);
            int digit = code % 3;
            code /= 3;
            s = digit == 0 ? ta : digit == 1 ? tb : tc;
        }
        out[base + k] = s;
    }
}

// ---------- fill_triangle kernel ----------
// Mirrors acc.exec((width, 1, 1), (32, 1, 1)) from AnyDSL fill_triangle, one
// row of the nd_range per triangle that reached limit <= 0.

void fill_triangle_kernel(Point a, Point b, Point c,
                                     uint8_t* pixels, int width, int height,
                                     float x_min, float x_max, float y_min, float y_max) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
    int px = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
             item_ct1.get_local_id(1);
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

// ---------- Background fill ----------

void init_background(uint8_t* pixels, int width, int height) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int y = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);
    if (x >= width || y >= height) return;
    uint8_t* p = pixels + (y * width + x) * 3;
    p[0] = 0; p[1] = 255; p[2] = 0;
}

// ---------- Host entry point ----------

extern "C" void make_image(uint8_t *buf, int width, int height, double x_min,
                           double x_max, double y_min, double y_max) {
dpct::device_ext &dev_ct1 = dpct::get_current_device();
sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    size_t size = (size_t)width * height * 3;

    uint8_t* d_pixels = sycl::malloc_device<uint8_t>(size, q_ct1);

    // Fill background green — mirrors acc.exec((width, height, 1), (8, 8, 1)).
    dpct::dim3 bg_block(8, 8);
    dpct::dim3 bg_grid((width + 7) / 8, (height + 7) / 8);
    q_ct1.parallel_for(sycl::nd_range<3>(bg_grid * bg_block, bg_block),
                       [=](sycl::nd_item<3> item_ct1) {
                           init_background(d_pixels, width, height);
                       });

    float fx_min = (float)x_min, fx_max = (float)x_max;
    float fy_min = (float)y_min, fy_max = (float)y_max;

    // Mirrors render_entry: the recursion starts from one triangle at limit 20.
    Triangle root = {{0.0f, 0.0f}, {-1.0f, 1.414f}, {1.0f, 1.414f}};
    Triangle* d_tris = sycl::malloc_device<Triangle>(1, q_ct1);
    q_ct1.memcpy(d_tris, &root, sizeof(Triangle));
    uint32_t* d_count = sycl::malloc_device<uint32_t>(1, q_ct1);

    size_t num_tris = 1;
    int limit = 20;
    while (num_tris > 0) {
        int depth = std::min(limit, 3);
        size_t num_children = depth == 3 ? 27 : depth == 2 ? 9 : depth == 1 ? 3 : 1;
        Triangle* d_next = sycl::malloc_device<Triangle>(num_tris * num_children, q_ct1);

        q_ct1.memset(d_count, 0, sizeof(uint32_t));
        q_ct1.parallel_for(sycl::range<1>(num_tris), [=](sycl::id<1> i) {
            expand_triangle(d_tris[i], depth, d_next, d_count, d_pixels,
                            width, height, fx_min, fx_max, fy_min, fy_max);
        });
        uint32_t count;
        q_ct1.memcpy(&count, d_count, sizeof(uint32_t)).wait();

        sycl::free(d_tris, q_ct1);
        d_tris = d_next;
        num_tris = count;

        if (depth > 0) {
            limit -= depth;
            continue;
        }

        // limit <= 0: every remaining triangle gets filled.
        if (num_tris > 0) {
            Triangle* tris = d_tris;
            q_ct1.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(num_tris, (width + 31) / 32 * 32),
                                  sycl::range<2>(1, 32)),
                [=](sycl::nd_item<2> item_ct1) {
                    Triangle t = tris[item_ct1.get_global_id(0)];
                    fill_triangle_kernel(t.a, t.b, t.c, d_pixels, width, height,
                                         fx_min, fx_max, fy_min, fy_max);
                });
        }
        break;
    }

    q_ct1.memcpy(buf, d_pixels, size).wait();
    sycl::free(d_tris, q_ct1);
    sycl::free(d_count, q_ct1);
    dpct::dpct_free(d_pixels, q_ct1);
}
