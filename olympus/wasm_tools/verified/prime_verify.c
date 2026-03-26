/*
 * VERIFIER: Check a primality claim.
 * Input: "N CLAIM [FACTOR]"
 *   CLAIM=1 means "N is prime"
 *   CLAIM=0 FACTOR means "N is composite, FACTOR divides it"
 *
 * If CLAIM=1 (prime): exhaustively verify no d in [2, sqrt(N)] divides N
 * If CLAIM=0 (composite): verify FACTOR divides N AND FACTOR != 1 AND FACTOR != N
 *
 * Output: "VALID" or "INVALID reason"
 */
__attribute__((noinline, optnone))
void compute(const char *input) {
    int N, claim, factor;
    factor = 0;
    sscanf(input, "%d %d %d", &N, &claim, &factor);

    if (N < 2) {
        if (claim == 0) {
            print_str("VALID trivial_composite");
        } else {
            print_str("INVALID less_than_2_not_prime");
        }
        putchar(10);
        return;
    }

    if (claim == 1) {
        /* Verify prime: check all divisors up to sqrt(N) */
        int d;
        for (d = 2; d * d <= N; d++) {
            if (N % d == 0) {
                print_str("INVALID claimed_prime_but_divisible_by_");
                print_int(d);
                putchar(10);
                return;
            }
        }
        print_str("VALID prime_exhaustively_verified");
        putchar(10);
    } else {
        /* Verify composite: check factor is valid */
        if (factor < 2 || factor >= N) {
            print_str("INVALID trivial_factor");
            putchar(10);
            return;
        }
        if (N % factor != 0) {
            print_str("INVALID factor_does_not_divide");
            putchar(10);
            return;
        }
        /* Factor is valid witness of compositeness */
        print_str("VALID composite_witness_");
        print_int(factor);
        putchar(10);
    }
}
