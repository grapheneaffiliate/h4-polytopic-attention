/*
 * NOVEL: Mobius Autocorrelation — RH Diagnostic
 *
 * C(k) = sum_{n=1}^{N-k} mu(n) * mu(n+k)
 *
 * Under RH, mu(n) behaves pseudo-randomly on squarefree integers.
 * The autocorrelation decay rate is a novel diagnostic for RH:
 *   - Fast decay (~k^{-1/2}): consistent with random-like mu => RH
 *   - Slow/no decay: structural correlations constraining zeros
 *
 * Input: "N K" — compute C(1)..C(K) for sum over n=1..N
 * Output: C(k) values
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
    int N, K;
    sscanf(input, "%d %d", &N, &K);
    if (N < 2) N = 2;
    if (K < 1) K = 1;
    if (K > 30) K = 30;

    int k;
    for (k = 1; k <= K; k++) {
        int ck = 0;
        int n;
        for (n = 1; n + k <= N; n++) {
            int mu_n = mobius(n);
            if (mu_n == 0) continue;
            int mu_nk = mobius(n + k);
            ck += mu_n * mu_nk;
        }
        print_str("C ");
        print_int(k);
        putchar(32);
        print_int(ck);
        putchar(10);
    }
}
