/*
 * VERIFIER: Check that a sort output is correct.
 * Input: "N a1..aN | b1..bN"
 *   Left of |: original array
 *   Right of |: claimed sorted array
 *
 * Checks TWO properties:
 *   1. Output is sorted (b[i] <= b[i+1] for all i)
 *   2. Output is a permutation of input (same multiset)
 *
 * Output: "VALID" or "INVALID reason"
 */
__attribute__((noinline, optnone))
void compute(const char *input) {
    int orig[64], sorted[64];
    int n_orig = 0, n_sorted = 0;
    int section = 0; /* 0 = before |, 1 = after | */
    int val = 0, neg = 0, has_val = 0;
    int i;

    for (i = 0; input[i] != 0; i++) {
        if (input[i] == '|') {
            if (has_val) {
                if (neg) val = -val;
                if (section == 0) { orig[n_orig] = val; n_orig++; }
                else { sorted[n_sorted] = val; n_sorted++; }
                val = 0; neg = 0; has_val = 0;
            }
            section = 1;
        } else if (input[i] >= '0' && input[i] <= '9') {
            val = val * 10 + (input[i] - '0');
            has_val = 1;
        } else if (input[i] == '-') {
            neg = 1;
        } else if (has_val) {
            if (neg) val = -val;
            if (section == 0) { orig[n_orig] = val; n_orig++; }
            else { sorted[n_sorted] = val; n_sorted++; }
            val = 0; neg = 0; has_val = 0;
        }
    }
    if (has_val) {
        if (neg) val = -val;
        if (section == 0) { orig[n_orig] = val; n_orig++; }
        else { sorted[n_sorted] = val; n_sorted++; }
    }

    /* Check 0: same length */
    if (n_orig != n_sorted) {
        print_str("INVALID length_mismatch");
        putchar(10);
        return;
    }

    /* Check 1: sorted order */
    for (i = 0; i < n_sorted - 1; i++) {
        if (sorted[i] > sorted[i + 1]) {
            print_str("INVALID not_sorted_at_");
            print_int(i);
            putchar(10);
            return;
        }
    }

    /* Check 2: permutation (same multiset) */
    /* For each element in orig, find and "use" a matching element in sorted */
    int used[64];
    for (i = 0; i < n_sorted; i++) used[i] = 0;

    int j;
    for (i = 0; i < n_orig; i++) {
        int found = 0;
        for (j = 0; j < n_sorted; j++) {
            if (!used[j] && sorted[j] == orig[i]) {
                used[j] = 1;
                found = 1;
                break;
            }
        }
        if (!found) {
            print_str("INVALID not_permutation_missing_");
            print_int(orig[i]);
            putchar(10);
            return;
        }
    }

    print_str("VALID sorted permutation_verified");
    putchar(10);
}
