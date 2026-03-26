/*
 * TVM-VERIFIED LEDGER: Verify account state integrity
 *
 * Given N account balances, verifies:
 *   1. No negative balances
 *   2. Total supply equals expected supply
 *   3. All balances are within i32 range (no overflow)
 *
 * Input: "expected_supply N bal1 bal2 ... balN"
 * Output: "VALID supply=X accounts=N" or "INVALID reason"
 */

__attribute__((noinline, optnone))
void compute(const char *input) {
    int pos = 0;
    int expected_supply = 0;
    int n = 0;
    int balances[256];

    /* Parse expected supply */
    while (input[pos] == ' ') pos++;
    while (input[pos] >= '0' && input[pos] <= '9') {
        int d = input[pos] - '0';
        int t2 = expected_supply + expected_supply;
        int t4 = t2 + t2;
        int t8 = t4 + t4;
        expected_supply = t8 + t2 + d;
        pos++;
    }

    /* Parse N */
    while (input[pos] == ' ') pos++;
    while (input[pos] >= '0' && input[pos] <= '9') {
        int d = input[pos] - '0';
        int t2 = n + n;
        int t4 = t2 + t2;
        int t8 = t4 + t4;
        n = t8 + t2 + d;
        pos++;
    }

    if (n > 256) {
        print_str("INVALID too_many_accounts");
        putchar(10);
        return;
    }

    /* Parse balances */
    int i;
    for (i = 0; i < n; i++) {
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
        balances[i] = v;
    }

    /* Check 1: no negative balances */
    for (i = 0; i < n; i++) {
        if (balances[i] < 0) {
            print_str("INVALID negative_balance_account_");
            print_int(i);
            putchar(10);
            return;
        }
    }

    /* Check 2: total supply matches expected */
    int total = 0;
    for (i = 0; i < n; i++) {
        total += balances[i];
        /* Check for overflow */
        if (total < 0) {
            print_str("INVALID supply_overflow");
            putchar(10);
            return;
        }
    }

    if (total != expected_supply) {
        print_str("INVALID supply_mismatch_expected_");
        print_int(expected_supply);
        print_str("_got_");
        print_int(total);
        putchar(10);
        return;
    }

    print_str("VALID supply=");
    print_int(total);
    print_str(" accounts=");
    print_int(n);
    putchar(10);
}
