/*
 * SOLVER: Find longest increasing subsequence length
 * Input: "N a1 a2 ... aN"
 * Output: "LEN i1 i2 ... iLEN" (length + indices of the subsequence)
 */
__attribute__((noinline, optnone))
void compute(const char *input) {
    int arr[64], dp[64], prev[64], seq[64];
    int n = 0;
    int val = 0, neg = 0, has_val = 0;
    int i;

    for (i = 0; input[i] != 0; i++) {
        if (input[i] >= '0' && input[i] <= '9') {
            val = val * 10 + (input[i] - '0');
            has_val = 1;
        } else if (input[i] == '-') {
            neg = 1;
        } else if (has_val) {
            if (neg) val = -val;
            arr[n] = val;
            n++;
            val = 0; neg = 0; has_val = 0;
        }
    }
    if (has_val) { if (neg) val = -val; arr[n] = val; n++; }

    /* DP: dp[i] = length of LIS ending at i */
    int best = 0, best_idx = 0;
    for (i = 0; i < n; i++) {
        dp[i] = 1;
        prev[i] = -1;
        int j;
        for (j = 0; j < i; j++) {
            if (arr[j] < arr[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                prev[i] = j;
            }
        }
        if (dp[i] > best) { best = dp[i]; best_idx = i; }
    }

    /* Reconstruct */
    int len = best;
    int idx = best_idx;
    for (i = len - 1; i >= 0; i--) {
        seq[i] = idx;
        idx = prev[idx];
    }

    /* Output: length then indices */
    print_int(len);
    for (i = 0; i < len; i++) {
        putchar(32);
        print_int(seq[i]);
    }
    putchar(10);
}
