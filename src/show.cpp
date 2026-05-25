#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include <raylib.h>
#include "results.hpp"

// ── Helpers ────────────────────────────────────────────────────────────────────

static Texture2D make_texture(const std::vector<float>& pixels) {
    Image img = GenImageColor(28, 28, BLACK);
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            unsigned char v = (unsigned char)(pixels[(size_t)(y * 28 + x)] * 255.0f);
            Color c = {v, v, v, 255};
            ImageDrawPixel(&img, x, y, c);
        }
    }
    Texture2D t = LoadTextureFromImage(img);
    UnloadImage(img);
    return t;
}

// Card border color: green=robust, red=attacked, yellow=unverified
static Color verdict_color(const ImageResult& r) {
    if (r.robust)   return {50,  220, 80,  255};  // green
    if (r.attacked) return {220, 60,  60,  255};  // red
    return              {200, 180, 40,  255};      // yellow
}

// ── Layout constants ───────────────────────────────────────────────────────────
static constexpr int CARD_W  = 190;
static constexpr int CARD_H  = 115;
static constexpr int GAP     = 8;
static constexpr int COLS    = 6;
static constexpr int HEADER  = 54;
static constexpr int WIN_W   = COLS * (CARD_W + GAP) + GAP;   // 1192
static constexpr int WIN_H   = 760;
static constexpr int IMG_SZ  = 56;   // 28 × 2 scale

int main() {
    auto results = load_results("output/results.bin");
    if (results.empty()) {
        fprintf(stderr,
            "output/results.bin not found.\n"
            "Run  ./attack  then  ./verify  first.\n");
        return 1;
    }

    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(WIN_W, WIN_H, "DeepPoly MNIST Robustness Demo");
    SetTargetFPS(60);

    // Load all textures up-front
    std::vector<Texture2D> orig_tex(results.size());
    std::vector<Texture2D> adv_tex (results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        orig_tex[i] = make_texture(results[i].orig_pixels);
        adv_tex [i] = make_texture(results[i].adv_pixels);
    }

    // Pre-compute summary stats (static)
    int n_attacked = 0, n_robust = 0, n_correct = 0;
    for (const auto& r : results) {
        if (r.attacked)               ++n_attacked;
        if (r.robust)                 ++n_robust;
        if (r.pred_clean==r.true_label) ++n_correct;
    }
    int total = (int)results.size();

    int scroll     = 0;
    int total_rows = (total + COLS - 1) / COLS;
    int content_h  = total_rows * (CARD_H + GAP) + GAP;

    while (!WindowShouldClose()) {
        // ── Input ────────────────────────────────────────────────────────────
        scroll -= (int)(GetMouseWheelMove() * 35.0f);
        scroll  = std::clamp(scroll, 0, std::max(0, content_h - (WIN_H - HEADER)));

        // ── Draw ─────────────────────────────────────────────────────────────
        BeginDrawing();
        ClearBackground({18, 18, 26, 255});

        // Header bar
        DrawRectangle(0, 0, WIN_W, HEADER, {28, 28, 40, 255});
        DrawRectangleLines(0, HEADER - 1, WIN_W, 1, {60, 60, 80, 255});

        DrawText("DeepPoly MNIST", 10, 10, 22, WHITE);

        char stats[256];
        snprintf(stats, sizeof(stats),
                 "%d imgs  |  %d correct (%.0f%%)  |  "
                 "%d attacked (%.0f%%)  |  %d certified robust (%.0f%%)",
                 total,
                 n_correct,  100.f * n_correct  / total,
                 n_attacked, 100.f * n_attacked / total,
                 n_robust,   100.f * n_robust   / total);
        DrawText(stats, 10, 34, 13, LIGHTGRAY);

        // Legend
        DrawRectangle(WIN_W - 220, 10, 12, 12, {50, 220, 80, 255});
        DrawText("Robust",    WIN_W - 205, 10, 13, LIGHTGRAY);
        DrawRectangle(WIN_W - 155, 10, 12, 12, {220, 60, 60, 255});
        DrawText("Attacked",  WIN_W - 140, 10, 13, LIGHTGRAY);
        DrawRectangle(WIN_W - 80,  10, 12, 12, {200, 180, 40, 255});
        DrawText("Unverif.", WIN_W - 65, 10, 13, LIGHTGRAY);

        // Cards
        for (int i = 0; i < total; ++i) {
            int col = i % COLS;
            int row = i / COLS;
            int cx  = GAP + col * (CARD_W + GAP);
            int cy  = HEADER + GAP + row * (CARD_H + GAP) - scroll;

            // Cull off-screen cards
            if (cy + CARD_H < HEADER || cy > WIN_H) continue;

            const auto& r = results[(size_t)i];
            Color border  = verdict_color(r);

            // Card background + border
            DrawRectangle(cx,     cy,     CARD_W,     CARD_H,     border);
            DrawRectangle(cx + 2, cy + 2, CARD_W - 4, CARD_H - 4, {28, 28, 40, 255});

            // Original image (left, 2× scale → 56×56)
            DrawTextureEx(orig_tex[(size_t)i],
                          {(float)(cx + 4), (float)(cy + 4)},
                          0.0f, 2.0f, WHITE);

            // Adversarial image (right of original, gap=6)
            DrawTextureEx(adv_tex[(size_t)i],
                          {(float)(cx + 4 + IMG_SZ + 6), (float)(cy + 4)},
                          0.0f, 2.0f, WHITE);

            // Captions under images
            DrawText("orig", cx + 4,              cy + IMG_SZ + 6,  11, GRAY);
            DrawText("adv",  cx + 4 + IMG_SZ + 6, cy + IMG_SZ + 6,  11, GRAY);

            // Right-side label column
            int lx = cx + 4 + IMG_SZ + 6 + IMG_SZ + 4;   // after both images
            Color clean_col = (r.pred_clean == r.true_label) ? GREEN : RED;
            Color adv_col   = r.attacked ? RED : GREEN;
            DrawText(TextFormat("t=%d",  r.true_label), lx, cy + 8,  12, LIGHTGRAY);
            DrawText(TextFormat("c=%d",  r.pred_clean), lx, cy + 22, 12, clean_col);
            DrawText(TextFormat("a=%d",  r.pred_adv),   lx, cy + 36, 12, adv_col);

            // Margin (small, bottom-right of label area)
            DrawText(TextFormat("%.3f", r.margin), lx, cy + 52, 10,
                     r.robust ? GREEN : GRAY);

            // Verdict at bottom of card
            const char* vtext = r.robust   ? "ROBUST"    :
                                r.attacked ? "ATTACKED"  : "UNVERIFIED";
            DrawText(vtext, cx + 4, cy + CARD_H - 14, 11, border);
        }

        // Scrollbar hint
        if (content_h > WIN_H - HEADER) {
            float ratio = (float)(WIN_H - HEADER) / (float)content_h;
            int bar_h   = std::max(20, (int)((WIN_H - HEADER) * ratio));
            int bar_y   = HEADER + (int)((float)(WIN_H - HEADER - bar_h) *
                          ((float)scroll / (float)(content_h - (WIN_H - HEADER))));
            DrawRectangle(WIN_W - 6, bar_y, 4, bar_h, {80, 80, 120, 200});
        }

        EndDrawing();
    }

    for (auto& t : orig_tex) UnloadTexture(t);
    for (auto& t : adv_tex)  UnloadTexture(t);
    CloseWindow();
    return 0;
}
