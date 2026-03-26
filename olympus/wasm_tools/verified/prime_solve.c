/*
 * SOLVER: Check primality of N
 * Input: "N"
 * Output: "1" if prime, "0" if composite, and if composite the smallest factor
 */
__attribute__((noinline, optnone))
void compute(const char *input) {
    int n;
    sscanf(input, "%d", &n);

    if (n < 2) {
        print_str("0 trivial");
        putchar(10);
        return;
    }

    int d;
    for (d = 2; d * d <= n; d++) {
        if (n % d == 0) {
            print_str("0 ");
            print_int(d);
            putchar(10);
            return;
        }
    }
    print_str("1");
    putchar(10);
}
