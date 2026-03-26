/*
 * TVM CHECKPOINT VERIFIER
 *
 * Verifies a prime count for a small range.
 * Input: "START END CLAIMED_COUNT"
 *   Counts primes in [START, END] via trial division.
 *   Compares against CLAIMED_COUNT.
 *
 * Output: "VALID count=X" or "INVALID expected=X got=Y"
 */

__attribute__((noinline, optnone))
int is_prime(int n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    int d;
    for (d = 3; d * d <= n; d += 2) {
        if (n % d == 0) return 0;
    }
    return 1;
}

void compute(const char *input) {
    int start, end, claimed;
    /* Parse three integers manually */
    int pos = 0;
    int vals[3];
    int vi = 0;

    while (vi < 3) {
        int v = 0;
        int neg = 0;
        while (input[pos] == ' ') pos++;
        if (input[pos] == '-') { neg = 1; pos++; }
        while (input[pos] >= '0' && input[pos] <= '9') {
            int d = input[pos] - '0';
            int t2 = v + v;
            int t4 = t2 + t2;
            int t8 = t4 + t4;
            v = t8 + t2 + d;
            pos++;
        }
        if (neg) v = -v;
        vals[vi] = v;
        vi++;
    }
    start = vals[0];
    end = vals[1];
    claimed = vals[2];

    int actual = 0;
    int n;
    for (n = start; n <= end; n++) {
        if (is_prime(n)) {
            actual++;
        }
    }

    if (actual == claimed) {
        print_str("VALID count=");
        print_int(actual);
    } else {
        print_str("INVALID expected=");
        print_int(actual);
        print_str(" got=");
        print_int(claimed);
    }
    putchar(10);
}
