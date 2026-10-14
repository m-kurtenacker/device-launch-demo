/* Basicly all of this is vibe coded using claude. */

#include <SDL2/SDL.h>
#include <cstdint>
#include <cmath>
#include <vector>

static const int WIDTH  = 1600;
static const int HEIGHT = 1200;

extern "C" void make_image(uint8_t* buf, int width, int height,
                double x_min, double x_max, double y_min, double y_max);

int main(int /*argc*/, char* /*argv*/[]) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        return 1;

    SDL_Window*   window   = SDL_CreateWindow(
        "Mandelbrot", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIDTH, HEIGHT, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture*  texture  = SDL_CreateTexture(
        renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

    std::vector<uint8_t> buf(WIDTH * HEIGHT * 3);

    // Current view in complex-plane coordinates
    double x_min = -2.5, x_max = 1.0;
    double y_min = -1.25, y_max = 1.25;
    bool   dirty = true;

    // Drag-pan state: store the view at the moment the drag started
    bool   panning        = false;
    int    drag_start_px  = 0, drag_start_py = 0;
    double drag_x_min     = 0, drag_x_max = 0;
    double drag_y_min     = 0, drag_y_max = 0;

    bool running = true;
    SDL_Event ev;
    while (running) {
        dirty = true;
        while (SDL_PollEvent(&ev)) {
            switch (ev.type) {
            case SDL_QUIT:
                running = false;
                break;

            case SDL_KEYDOWN:
                if (ev.key.keysym.sym == SDLK_ESCAPE) running = false;
                break;

            case SDL_MOUSEWHEEL: {
                // Zoom centered on the current mouse position
                int mx, my;
                SDL_GetMouseState(&mx, &my);
                double cx    = x_min + (x_max - x_min) * mx / WIDTH;
                double cy    = y_min + (y_max - y_min) * my / HEIGHT;
                float  delta = (ev.wheel.direction == SDL_MOUSEWHEEL_FLIPPED)
                               ? -ev.wheel.preciseY : ev.wheel.preciseY;
                double factor = 1 - (delta * 0.25);
                x_min = cx + (x_min - cx) * factor;
                x_max = cx + (x_max - cx) * factor;
                y_min = cy + (y_min - cy) * factor;
                y_max = cy + (y_max - cy) * factor;
                dirty = true;
                break;
            }

            case SDL_MOUSEBUTTONDOWN:
                if (ev.button.button == SDL_BUTTON_LEFT) {
                    panning       = true;
                    drag_start_px = ev.button.x;
                    drag_start_py = ev.button.y;
                    drag_x_min    = x_min;  drag_x_max = x_max;
                    drag_y_min    = y_min;  drag_y_max = y_max;
                }
                break;

            case SDL_MOUSEBUTTONUP:
                if (ev.button.button == SDL_BUTTON_LEFT)
                    panning = false;
                break;

            case SDL_MOUSEMOTION:
                if (panning) {
                    // Map pixel delta to complex-plane delta and shift the view
                    double dx = (drag_start_px - ev.motion.x)
                                * (drag_x_max - drag_x_min) / WIDTH;
                    double dy = (drag_start_py - ev.motion.y)
                                * (drag_y_max - drag_y_min) / HEIGHT;
                    x_min = drag_x_min + dx;  x_max = drag_x_max + dx;
                    y_min = drag_y_min + dy;  y_max = drag_y_max + dy;
                    dirty = true;
                }
                break;
            }
        }

        if (dirty) {
            make_image(buf.data(), WIDTH, HEIGHT, x_min, x_max, y_min, y_max);
            SDL_UpdateTexture(texture, nullptr, buf.data(), WIDTH * 3);
            dirty = false;
        }

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
        //SDL_Delay(16);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
