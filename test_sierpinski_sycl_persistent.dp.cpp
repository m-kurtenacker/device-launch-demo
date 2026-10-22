#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>

namespace syclex = sycl::ext::oneapi::experimental;

// Emulation of CUDA Dynamic Parallelism (CDP) with a persistent kernel.
//
// SYCL cannot launch kernels from device code. Instead, a single persistent
// kernel is started with as many work-groups as can be resident on the device
// at once. A device-side "launch" pushes a descriptor onto a global queue, one
// entry per block of the launch grid; every work-group acts like a CUDA block
// that repeatedly pops an entry and executes it, with its local id in the role
// of threadIdx.x. The kernel finishes once no launched block is pending.

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

// ---------- Device-side launch queue ----------

// Work-group size of the persistent kernel, i.e. the largest block that can be
// launched from the device: 32 for fill_triangle, 27 for render_kernel_27.
static constexpr int BLOCK_SIZE = 32;

enum KernelId : int { RENDER_KERNEL, FILL_TRIANGLE_KERNEL };

// One block of a device-side kernel launch.
//
// RENDER_KERNEL stands for render_kernel_3/9/27 (and render_entry): a single
// block of 3^depth threads, where thread i renders the i-th sub-triangle of
// `tri` after `depth` levels of subdivision. CUDA passes the precomputed batch
// by value; here each thread recomputes its sub-triangle so that queue entries
// stay small.
// FILL_TRIANGLE_KERNEL is one block (block_idx) of fill_triangle_kernel.
struct Launch {
    Triangle tri;
    int kernel;
    int limit;
    int depth;
    int block_idx;
};

struct QueueState {
    uint32_t head;      // next queue slot to be executed
    uint32_t tail;      // next free queue slot
    uint32_t pending;   // launched blocks that have not finished executing
    uint32_t overflow;  // set if a launch did not fit into the queue
};

struct LaunchQueue {
    Launch* entries;
    uint32_t* ready;    // ready[i] != 0 once entries[i] has been written
    uint32_t capacity;
    QueueState* state;
};

template <typename T>
using device_atomic = sycl::atomic_ref<T, sycl::memory_order::acq_rel,
                                       sycl::memory_scope::device,
                                       sycl::access::address_space::global_space>;

// Device-side equivalent of kernel<<<num_blocks, ...>>>(...).
__dpct_inline__ void device_launch(const LaunchQueue &queue, Launch launch,
                                   int num_blocks) {
    device_atomic<uint32_t> pending(queue.state->pending);
    device_atomic<uint32_t> tail(queue.state->tail);
    for (int block = 0; block < num_blocks; block++) {
        pending.fetch_add(1);
        uint32_t slot = tail.fetch_add(1);
        if (slot >= queue.capacity) {
            device_atomic<uint32_t>(queue.state->overflow).store(1);
            pending.fetch_sub(1);
            continue;
        }
        launch.block_idx = block;
        queue.entries[slot] = launch;
        device_atomic<uint32_t>(queue.ready[slot]).store(1);
    }
}

// ---------- Core render function ----------
//
// Same as the CUDA render_triangle, with every CDP launch (<<<>>> from device
// code) replaced by device_launch.

void render_triangle(Point a, Point b, Point c, int limit,
                     const LaunchQueue &queue, uint8_t *pixels, int width,
                     int height, float x_min, float x_max, float y_min,
                     float y_max) {
    if (offscreen(a, b, c, x_min, x_max, y_min, y_max))
        return;

    if (within_one_pixel(a, b, c, width, height, x_min, x_max, y_min, y_max)) {
        draw_point(a, pixels, width, height, x_min, x_max, y_min, y_max);
        return;
    }

    if (limit <= 0) {
        // acc.exec((width, 1, 1), (32, 1, 1)) from fill_triangle.
        device_launch(queue, {{a, b, c}, FILL_TRIANGLE_KERNEL, 0, 0, 0},
                      (width + BLOCK_SIZE - 1) / BLOCK_SIZE);
        return;
    }

    if (limit >= 3) {
        // acc.exec((27, 1, 1), (27, 1, 1)) — expand three Sierpinski levels at once.
        device_launch(queue, {{a, b, c}, RENDER_KERNEL, limit - 3, 3, 0}, 1);
        return;
    }

    if (limit >= 2) {
        // acc.exec((9, 1, 1), (9, 1, 1)) — expand two levels at once.
        device_launch(queue, {{a, b, c}, RENDER_KERNEL, limit - 2, 2, 0}, 1);
        return;
    }

    // limit == 1: acc.exec((3, 1, 1), (3, 1, 1)) — one subdivision level.
    device_launch(queue, {{a, b, c}, RENDER_KERNEL, limit - 1, 1, 0}, 1);
}

// ---------- Emulated kernels ----------

// render_kernel_3/9/27: thread `id` renders sub-triangle `id`, found by
// following the base-3 digits of id down `depth` levels of split_triangle.
void render_kernel(int id, const Launch &launch, const LaunchQueue &queue,
                   uint8_t *pixels, int width, int height, float x_min,
                   float x_max, float y_min, float y_max) {
    int num_threads = launch.depth == 3 ? 27 : launch.depth == 2 ? 9
                    : launch.depth == 1 ? 3 : 1;
    if (id >= num_threads) return;

    Triangle t = launch.tri;
    int code = id;
    for (int l = 0; l < launch.depth; l++) {
        Triangle ta, tb, tc;
        split_triangle(t, ta, tb, tc);
        int digit = code % 3;
        code /= 3;
        t = digit == 0 ? ta : digit == 1 ? tb : tc;
    }
    render_triangle(t.a, t.b, t.c, launch.limit, queue, pixels, width, height,
                    x_min, x_max, y_min, y_max);
}

// fill_triangle_kernel: mirrors acc.exec((width, 1, 1), (32, 1, 1)) from
// AnyDSL fill_triangle.
void fill_triangle_kernel(int id, const Launch &launch, uint8_t *pixels,
                          int width, int height, float x_min, float x_max,
                          float y_min, float y_max) {
    Point a = launch.tri.a, b = launch.tri.b, c = launch.tri.c;
    int px = launch.block_idx * BLOCK_SIZE + id;
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

// ---------- Persistent scheduler kernel ----------

class sierpinski_scheduler;

void scheduler(sycl::nd_item<1> item, Launch &current, int &running,
               const LaunchQueue &queue, uint8_t *pixels, int width,
               int height, float x_min, float x_max, float y_min, float y_max) {
    int lid = item.get_local_id(0);
    device_atomic<uint32_t> head(queue.state->head);
    device_atomic<uint32_t> pending(queue.state->pending);

    while (true) {
        // The group leader claims the next slot and waits until it has been
        // written. A slot that is never written (beyond the last launch) is
        // detected by all launched blocks having finished.
        if (lid == 0) {
            uint32_t slot = head.fetch_add(1);
            while (true) {
                if (slot < queue.capacity &&
                    device_atomic<uint32_t>(queue.ready[slot]).load()) {
                    current = queue.entries[slot];
                    running = 1;
                    break;
                }
                if (pending.load() == 0) {
                    running = 0;
                    break;
                }
            }
        }
        sycl::group_barrier(item.get_group());
        if (!running)
            break;

        Launch launch = current;
        if (launch.kernel == RENDER_KERNEL)
            render_kernel(lid, launch, queue, pixels, width, height,
                          x_min, x_max, y_min, y_max);
        else
            fill_triangle_kernel(lid, launch, pixels, width, height,
                                 x_min, x_max, y_min, y_max);

        // The block is done once all of its threads are; any blocks it
        // launched were counted in `pending` before this point.
        sycl::group_barrier(item.get_group());
        if (lid == 0)
            pending.fetch_sub(1);
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

// Number of work-groups that are guaranteed to be resident at the same time.
// The persistent kernel deadlocks if it is launched with more, since waiting
// groups could then keep the groups producing their work from being scheduled.
static size_t max_resident_work_groups(sycl::queue &q) {
    auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        q.get_context(), {q.get_device()},
        {sycl::get_kernel_id<sierpinski_scheduler>()});
    sycl::kernel kernel = bundle.get_kernel<sierpinski_scheduler>();
    size_t local_mem = sizeof(Launch) + sizeof(int);
    size_t num_groups = kernel.ext_oneapi_get_info<
        syclex::info::kernel_queue_specific::max_num_work_groups>(
        q, sycl::range<1>(BLOCK_SIZE), local_mem);
    return num_groups > 0 ? num_groups : 1;
}

extern "C" void make_image(uint8_t *buf, int width, int height, double x_min,
                           double x_max, double y_min, double y_max) {
dpct::device_ext &dev_ct1 = dpct::get_current_device();
sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    size_t size = (size_t)width * height * 3;

    static size_t num_groups = max_resident_work_groups(q_ct1);
    // Grown and kept across frames whenever the launches of a frame overflow it.
    static uint32_t capacity = 1 << 16;

    uint8_t* d_pixels = sycl::malloc_device<uint8_t>(size, q_ct1);
    QueueState* d_state = sycl::malloc_device<QueueState>(1, q_ct1);

    float fx_min = (float)x_min, fx_max = (float)x_max;
    float fy_min = (float)y_min, fy_max = (float)y_max;

    while (true) {
        LaunchQueue queue;
        queue.capacity = capacity;
        queue.entries = sycl::malloc_device<Launch>(capacity, q_ct1);
        queue.ready = sycl::malloc_device<uint32_t>(capacity, q_ct1);
        queue.state = d_state;

        // Fill background green — mirrors acc.exec((width, height, 1), (8, 8, 1)).
        dpct::dim3 bg_block(8, 8);
        dpct::dim3 bg_grid((width + 7) / 8, (height + 7) / 8);
        q_ct1.parallel_for(sycl::nd_range<3>(bg_grid * bg_block, bg_block),
                           [=](sycl::nd_item<3> item_ct1) {
                               init_background(d_pixels, width, height);
                           });

        // Mirrors render_entry<<<1, 1>>>: a single-thread render block for the
        // root triangle at limit 20 kicks off the launch tree.
        Triangle root = {{0.0f, 0.0f}, {-1.0f, 1.414f}, {1.0f, 1.414f}};
        Launch entry = {root, RENDER_KERNEL, 20, 0, 0};
        QueueState state = {0, 1, 1, 0};
        q_ct1.memset(queue.ready, 0, capacity * sizeof(uint32_t));
        q_ct1.memcpy(queue.entries, &entry, sizeof(Launch));
        q_ct1.memcpy(d_state, &state, sizeof(QueueState));
        q_ct1.fill(queue.ready, 1u, 1);

        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<Launch, 1> current(1, cgh);
            sycl::local_accessor<int, 1> running(1, cgh);
            syclex::nd_launch<sierpinski_scheduler>(
                cgh,
                syclex::launch_config(
                    sycl::nd_range<1>(num_groups * BLOCK_SIZE, BLOCK_SIZE),
                    syclex::properties{syclex::use_root_sync}),
                [=](sycl::nd_item<1> item) {
                    scheduler(item, current[0], running[0], queue, d_pixels,
                              width, height, fx_min, fx_max, fy_min, fy_max);
                });
        });
        q_ct1.memcpy(&state, d_state, sizeof(QueueState)).wait();

        sycl::free(queue.entries, q_ct1);
        sycl::free(queue.ready, q_ct1);
        if (!state.overflow)
            break;
        capacity *= 2;
    }

    q_ct1.memcpy(buf, d_pixels, size).wait();
    sycl::free(d_state, q_ct1);
    dpct::dpct_free(d_pixels, q_ct1);
}
