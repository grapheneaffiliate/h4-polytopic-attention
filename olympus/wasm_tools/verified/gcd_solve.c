/*
 * SOLVER: Compute GCD of two integers
 * Input: "a b"
 * Output: "G" where G = gcd(a, b)
 */
__attribute__((noinline, optnone))
void compute(const char *input) {
    int a, b;
    sscanf(input, "%d %d", &a, &b);
    if (a < 0) a = -a;
    if (b < 0) b = -b;

    int x = a, y = b;
    while (y != 0) {
        int t = y;
        y = x % y;
        x = t;
    }
    print_int(x);
    putchar(10);
}
