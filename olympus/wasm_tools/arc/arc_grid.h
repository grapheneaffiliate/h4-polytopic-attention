/*
 * ARC Grid Runtime for transformer-vm C programs.
 * Grid format: "H W v00 v01 ... v(H-1)(W-1)" (row-major, values 0-9).
 * Max grid: 30x30 = 900 cells.
 *
 * IMPORTANT: Avoids pointer-to-local parameters (WASM limitation).
 * Uses global state for grid parsing instead.
 */

#ifndef ARC_GRID_H
#define ARC_GRID_H

/* Global parse cursor — avoids needing pointer-to-int params */
static int _arc_pos;

__attribute__((always_inline))
static inline void arc_skip_spaces(const char *s) {
    while (s[_arc_pos] == ' ' || s[_arc_pos] == '\n' || s[_arc_pos] == '\t')
        _arc_pos++;
}

__attribute__((always_inline))
static inline int arc_parse_int(const char *s) {
    arc_skip_spaces(s);
    int neg = 0;
    if (s[_arc_pos] == '-') { neg = 1; _arc_pos++; }
    int val = 0;
    while (s[_arc_pos] >= '0' && s[_arc_pos] <= '9') {
        int d = s[_arc_pos] - '0';
        int t2 = val + val;
        int t4 = t2 + t2;
        int t8 = t4 + t4;
        val = t8 + t2 + d;
        _arc_pos++;
    }
    if (neg) val = 0 - val;
    return val;
}

/* Parse grid dimensions and values from input.
 * Sets grid[] (row-major), returns (h, w) via pointers to arrays.
 * h and w are stored in grid_h[0] and grid_w[0].
 * The caller must declare: int grid[900]; */
__attribute__((always_inline))
static inline void arc_parse_grid(const char *input, int *grid, int *h_out, int *w_out) {
    _arc_pos = 0;
    int h = arc_parse_int(input);
    int w = arc_parse_int(input);
    /* Store h, w via memory (arrays work, pointer-to-local doesn't) */
    h_out[0] = h;
    w_out[0] = w;
    int total = 0;
    int i;
    for (i = 0; i < h; i++) total = total + w;
    for (i = 0; i < total; i++) {
        grid[i] = arc_parse_int(input);
    }
}

/* Emit grid as "H W v0 v1 ..." followed by newline. */
__attribute__((always_inline))
static inline void arc_emit_grid(int *grid, int h, int w) {
    print_int(h);
    putchar(' ');
    print_int(w);
    int total = 0;
    int i;
    for (i = 0; i < h; i++) total = total + w;
    for (i = 0; i < total; i++) {
        putchar(' ');
        print_int(grid[i]);
    }
    putchar('\n');
}

/* Get cell at (row, col) in grid of width w. */
__attribute__((always_inline))
static inline int arc_get(int *grid, int w, int row, int col) {
    int offset = col;
    int i;
    for (i = 0; i < row; i++) offset = offset + w;
    return grid[offset];
}

/* Set cell at (row, col) in grid of width w. */
__attribute__((always_inline))
static inline void arc_set(int *grid, int w, int row, int col, int val) {
    int offset = col;
    int i;
    for (i = 0; i < row; i++) offset = offset + w;
    grid[offset] = val;
}

/* Fill entire grid with a single value. */
__attribute__((always_inline))
static inline void arc_fill(int *grid, int h, int w, int val) {
    int total = 0;
    int i;
    for (i = 0; i < h; i++) total = total + w;
    for (i = 0; i < total; i++) grid[i] = val;
}

/* Copy grid src to dst. */
__attribute__((always_inline))
static inline void arc_copy(int *dst, int *src, int h, int w) {
    int total = 0;
    int i;
    for (i = 0; i < h; i++) total = total + w;
    for (i = 0; i < total; i++) dst[i] = src[i];
}

#endif /* ARC_GRID_H */
