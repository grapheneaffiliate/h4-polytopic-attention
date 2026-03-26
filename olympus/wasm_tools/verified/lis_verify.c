/*
 * VERIFIER: Check an LIS (longest increasing subsequence) claim.
 * Input: "N a1..aN | LEN i1..iLEN"
 *   Left of |: original array
 *   Right of |: claimed LIS length + indices
 *
 * Checks FOUR properties:
 *   1. Indices are valid (0 <= i < N)
 *   2. Indices are strictly increasing (i[k] < i[k+1])
 *   3. Values at indices are strictly increasing (a[i[k]] < a[i[k+1]])
 *   4. No longer increasing subsequence exists (brute-force for small N)
 *
 * Output: "VALID" or "INVALID reason"
 */
__attribute__((noinline, optnone))
int lis_length_bf(int *arr, int n) {
    /* Brute force LIS via DP */
    int dp[64];
    int best = 0;
    int i, j;
    for (i = 0; i < n; i++) {
        dp[i] = 1;
        for (j = 0; j < i; j++) {
            if (arr[j] < arr[i] && dp[j] + 1 > dp[i])
                dp[i] = dp[j] + 1;
        }
        if (dp[i] > best) best = dp[i];
    }
    return best;
}

__attribute__((noinline, optnone))
void compute(const char *input) {
    int arr[64], indices[64];
    int n = 0, n_idx = 0, claimed_len = 0;
    int section = 0;
    int val = 0, neg = 0, has_val = 0;
    int first_after_pipe = 1;
    int i;

    for (i = 0; input[i] != 0; i++) {
        if (input[i] == '|') {
            if (has_val) {
                if (neg) val = -val;
                arr[n] = val; n++;
                val = 0; neg = 0; has_val = 0;
            }
            section = 1;
            first_after_pipe = 1;
        } else if (input[i] >= '0' && input[i] <= '9') {
            val = val * 10 + (input[i] - '0');
            has_val = 1;
        } else if (input[i] == '-') {
            neg = 1;
        } else if (has_val) {
            if (neg) val = -val;
            if (section == 0) { arr[n] = val; n++; }
            else if (first_after_pipe) { claimed_len = val; first_after_pipe = 0; }
            else { indices[n_idx] = val; n_idx++; }
            val = 0; neg = 0; has_val = 0;
        }
    }
    if (has_val) {
        if (neg) val = -val;
        if (section == 0) { arr[n] = val; n++; }
        else if (first_after_pipe) { claimed_len = val; first_after_pipe = 0; }
        else { indices[n_idx] = val; n_idx++; }
    }

    /* Check: claimed length matches index count */
    if (n_idx != claimed_len) {
        print_str("INVALID index_count_mismatch");
        putchar(10);
        return;
    }

    /* Check 1: valid indices */
    for (i = 0; i < n_idx; i++) {
        if (indices[i] < 0 || indices[i] >= n) {
            print_str("INVALID index_out_of_range");
            putchar(10);
            return;
        }
    }

    /* Check 2: indices strictly increasing */
    for (i = 0; i < n_idx - 1; i++) {
        if (indices[i] >= indices[i + 1]) {
            print_str("INVALID indices_not_increasing");
            putchar(10);
            return;
        }
    }

    /* Check 3: values strictly increasing */
    for (i = 0; i < n_idx - 1; i++) {
        if (arr[indices[i]] >= arr[indices[i + 1]]) {
            print_str("INVALID values_not_increasing");
            putchar(10);
            return;
        }
    }

    /* Check 4: optimality — no longer subsequence exists */
    int true_lis = lis_length_bf(arr, n);
    if (claimed_len < true_lis) {
        print_str("INVALID not_optimal_true_lis_");
        print_int(true_lis);
        putchar(10);
        return;
    }

    print_str("VALID increasing verified optimal");
    putchar(10);
}
