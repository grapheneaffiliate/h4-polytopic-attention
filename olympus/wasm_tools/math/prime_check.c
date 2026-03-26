/* Primality test by trial division.
 * Input: decimal integer, e.g. "97"
 * Output: "prime\n" or "composite factor=X\n"
 */

void compute(const char *input) {
    int n;
    sscanf(input, "%d", &n);

    if (n < 0) n = 0 - n;

    if (n < 2) {
        print_str("composite\n");
        return;
    }
    if (n == 2 || n == 3) {
        print_str("prime\n");
        return;
    }
    /* Check divisibility by 2 */
    int half = n / 2;
    if (half + half == n) {
        print_str("composite factor=2\n");
        return;
    }
    /* Check odd divisors up to sqrt(n) */
    int d = 3;
    while (d * d <= n) {
        int q = n / d;
        if (q * d == n) {
            print_str("composite factor=");
            print_int(d);
            putchar('\n');
            return;
        }
        d = d + 2;
    }
    print_str("prime\n");
}
