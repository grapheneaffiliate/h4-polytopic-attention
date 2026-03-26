/*
 * VERIFIER: Check that G is the GCD of A and B.
 * Input: "A B G"
 *
 * Checks THREE properties:
 *   1. G divides A  (A % G == 0)
 *   2. G divides B  (B % G == 0)
 *   3. G is maximal (no D > G divides both A and B)
 *
 * Property 3 is checked exhaustively: for D = G+1 to min(A,B),
 * verify D does NOT divide both A and B.
 *
 * Output: "VALID" or "INVALID reason"
 */
__attribute__((noinline, optnone))
void compute(const char *input) {
    int A, B, G;
    sscanf(input, "%d %d %d", &A, &B, &G);
    if (A < 0) A = -A;
    if (B < 0) B = -B;

    /* Check 1: G divides A */
    if (A % G != 0) {
        print_str("INVALID g_not_divides_a");
        putchar(10);
        return;
    }

    /* Check 2: G divides B */
    if (B % G != 0) {
        print_str("INVALID g_not_divides_b");
        putchar(10);
        return;
    }

    /* Check 3: maximality — no larger common divisor exists */
    int limit = A;
    if (B < limit) limit = B;

    int d;
    for (d = G + 1; d <= limit; d++) {
        if (A % d == 0 && B % d == 0) {
            print_str("INVALID larger_divisor_");
            print_int(d);
            putchar(10);
            return;
        }
    }

    print_str("VALID divides_both maximal");
    putchar(10);
}
