/*
 * Verified Robin's Inequality — sigma(n) for HCN
 *
 * Computes sigma(n) exactly for highly composite numbers.
 * Python does the Robin bound comparison (needs float).
 *
 * Input: n (single number to check)
 * Output: n, sigma(n), sigma(n)/n as reduced fraction
 */

__attribute__((noinline, optnone))
int sigma_func(int n) {
    if (n <= 0) return 0;
    int s = 0;
    int d;
    for (d = 1; d * d <= n; d++) {
        if (n % d == 0) {
            s += d;
            if (d != n / d) s += n / d;
        }
    }
    return s;
}

__attribute__((noinline, optnone))
int gcd(int a, int b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) { int t = b; b = a % b; a = t; }
    return a;
}

void compute(const char *input) {
    int n;
    sscanf(input, "%d", &n);
    if (n < 1) n = 1;

    int sig = sigma_func(n);
    int g = gcd(sig, n);

    print_str("S ");
    print_int(n);
    putchar(32);
    print_int(sig);
    putchar(32);
    print_int(sig / g);
    putchar(32);
    print_int(n / g);
    putchar(10);
}
