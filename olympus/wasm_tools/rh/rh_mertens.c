/*
 * Verified Mertens Function — TVM Exact Computation
 *
 * M(x) = sum_{n=1}^{x} mu(n) via trial factorization.
 * RH implies |M(x)| = O(x^{1/2+eps}).
 *
 * Input: N (upper limit)
 * Output: M(N), extremes, sign changes, M^2/x ratios
 */

__attribute__((noinline, optnone))
int mobius(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    int factors = 0;
    if (n % 2 == 0) { factors++; n = n / 2; if (n % 2 == 0) return 0; }
    int d;
    for (d = 3; d * d <= n; d += 2) {
        if (n % d == 0) { factors++; n = n / d; if (n % d == 0) return 0; }
    }
    if (n > 1) factors++;
    if (factors % 2 == 0) return 1;
    return -1;
}

void compute(const char *input) {
    int N;
    sscanf(input, "%d", &N);
    if (N < 1) N = 1;

    int M = 0;
    int max_M = 0, min_M = 0;
    int max_at = 1, min_at = 1;
    int sign_changes = 0;
    int prev_sign = 0;

    int x;
    for (x = 1; x <= N; x++) {
        M += mobius(x);

        int cur_sign = (M > 0) ? 1 : ((M < 0) ? -1 : 0);
        if (prev_sign != 0 && cur_sign != 0 && cur_sign != prev_sign)
            sign_changes++;
        if (cur_sign != 0) prev_sign = cur_sign;

        if (M > max_M) { max_M = M; max_at = x; }
        if (M < min_M) { min_M = M; min_at = x; }
    }

    /* Compact output for Python parsing */
    print_str("M ");
    print_int(N);
    putchar(32);
    print_int(M);
    putchar(10);
    print_str("MAX ");
    print_int(max_M);
    putchar(32);
    print_int(max_at);
    putchar(10);
    print_str("MIN ");
    print_int(min_M);
    putchar(32);
    print_int(min_at);
    putchar(10);
    print_str("SC ");
    print_int(sign_changes);
    putchar(10);
}
