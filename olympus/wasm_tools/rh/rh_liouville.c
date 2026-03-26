/*
 * Verified Liouville Summatory Function
 *
 * L(x) = sum_{n=1}^{x} lambda(n) where lambda(n) = (-1)^{Omega(n)}.
 * RH => L(x) = O(x^{1/2+eps}).
 *
 * Input: N
 * Output: L(N), extremes, Polya violation count
 */

__attribute__((noinline, optnone))
int omega_total(int n) {
    if (n <= 1) return 0;
    int count = 0;
    while (n % 2 == 0) { count++; n = n / 2; }
    int d;
    for (d = 3; d * d <= n; d += 2) {
        while (n % d == 0) { count++; n = n / d; }
    }
    if (n > 1) count++;
    return count;
}

void compute(const char *input) {
    int N;
    sscanf(input, "%d", &N);
    if (N < 1) N = 1;

    int L = 0;
    int max_L = 0, min_L = 0;
    int max_at = 1, min_at = 1;
    int sign_changes = 0;
    int prev_sign = 0;
    int polya_v = 0;

    int x;
    for (x = 1; x <= N; x++) {
        int omega = omega_total(x);
        L += (omega % 2 == 0) ? 1 : -1;

        int cur_sign = (L > 0) ? 1 : ((L < 0) ? -1 : 0);
        if (prev_sign != 0 && cur_sign != 0 && cur_sign != prev_sign)
            sign_changes++;
        if (cur_sign != 0) prev_sign = cur_sign;

        if (L > max_L) { max_L = L; max_at = x; }
        if (L < min_L) { min_L = L; min_at = x; }
        if (L > 0 && x >= 2) polya_v++;
    }

    print_str("L ");
    print_int(N);
    putchar(32);
    print_int(L);
    putchar(10);
    print_str("MAX ");
    print_int(max_L);
    putchar(32);
    print_int(max_at);
    putchar(10);
    print_str("MIN ");
    print_int(min_L);
    putchar(32);
    print_int(min_at);
    putchar(10);
    print_str("SC ");
    print_int(sign_changes);
    putchar(10);
    print_str("PV ");
    print_int(polya_v);
    putchar(10);
}
