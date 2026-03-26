/* Longest Increasing Subsequence — exact, O(n^2) DP with predecessor tracking.
 *
 * Input: space-separated integers, e.g. "10 9 2 5 3 7 101 18"
 * Output: the LIS as space-separated integers, e.g. "2 3 7 101"
 *
 * This is the algorithm the 3B specialist understands but can't implement:
 * correct DP table + correct predecessor backtracking.
 */

void compute(const char *input) {
    int arr[256];
    int n = 0;

    /* Parse space-separated integers */
    int pos = 0;
    while (input[pos]) {
        /* Skip spaces */
        while (input[pos] == ' ') pos++;
        if (input[pos] == 0) break;

        /* Parse number (possibly negative) */
        int neg = 0;
        if (input[pos] == '-') { neg = 1; pos++; }
        int val = 0;
        while (input[pos] >= '0' && input[pos] <= '9') {
            int d = input[pos] - '0';
            int t2 = val + val;
            int t4 = t2 + t2;
            int t8 = t4 + t4;
            val = t8 + t2 + d;
            pos++;
        }
        if (neg) val = 0 - val;
        arr[n] = val;
        n++;
    }

    if (n == 0) { putchar('\n'); return; }

    /* DP: dp[i] = length of LIS ending at index i */
    int dp[256];
    int prev[256];  /* predecessor index, -1 if none */
    int i, j;

    for (i = 0; i < n; i++) {
        dp[i] = 1;
        prev[i] = 0 - 1;  /* -1 */
    }

    for (i = 1; i < n; i++) {
        for (j = 0; j < i; j++) {
            if (arr[j] < arr[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                prev[i] = j;
            }
        }
    }

    /* Find the index with maximum LIS length */
    int max_len = 0;
    int max_idx = 0;
    for (i = 0; i < n; i++) {
        if (dp[i] > max_len) {
            max_len = dp[i];
            max_idx = i;
        }
    }

    /* Backtrack through predecessors to reconstruct the subsequence */
    int result[256];
    int rlen = 0;
    int idx = max_idx;
    while (idx >= 0) {
        result[rlen] = arr[idx];
        rlen++;
        idx = prev[idx];
    }

    /* Print in forward order (result is reversed) */
    for (i = rlen - 1; i >= 0; i--) {
        if (i < rlen - 1) putchar(' ');
        print_int(result[i]);
    }
    putchar('\n');
}
