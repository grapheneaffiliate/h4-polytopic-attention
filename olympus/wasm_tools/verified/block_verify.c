/*
 * TVM-VERIFIED BLOCK: Verify a batch of transfers + state transition
 *
 * Verifies an ENTIRE BLOCK of transactions in one pass:
 *   1. Each transfer has positive amount
 *   2. Each sender has sufficient balance at time of transfer
 *   3. No account goes negative at any point
 *   4. Final state matches claimed state
 *   5. Total supply is conserved across the entire block
 *
 * Input: "SUPPLY N_ACCOUNTS bal1..balN N_TX sender1 receiver1 amt1 ... final_bal1..final_balN"
 * Output: "VALID txs=X supply=Y" or "INVALID reason"
 *
 * This verifies the ENTIRE state transition of a block atomically.
 */

__attribute__((noinline, optnone))
int parse_int_at(const char *input, int *pos) {
    int v = 0;
    int neg = 0;
    while (input[*pos] == ' ') (*pos)++;
    if (input[*pos] == '-') { neg = 1; (*pos)++; }
    while (input[*pos] >= '0' && input[*pos] <= '9') {
        int d = input[*pos] - '0';
        int t2 = v + v;
        int t4 = t2 + t2;
        int t8 = t4 + t4;
        v = t8 + t2 + d;
        (*pos)++;
    }
    if (neg) v = -v;
    return v;
}

void compute(const char *input) {
    int pos = 0;
    int state[64];
    int final_state[64];

    /* Parse expected supply */
    int supply = parse_int_at(input, &pos);

    /* Parse number of accounts */
    int n_accts = parse_int_at(input, &pos);
    if (n_accts > 64 || n_accts < 1) {
        print_str("INVALID bad_account_count");
        putchar(10);
        return;
    }

    /* Parse initial balances */
    int i;
    int init_sum = 0;
    for (i = 0; i < n_accts; i++) {
        state[i] = parse_int_at(input, &pos);
        init_sum += state[i];
    }

    /* Verify initial supply */
    if (init_sum != supply) {
        print_str("INVALID init_supply_mismatch");
        putchar(10);
        return;
    }

    /* Parse number of transactions */
    int n_tx = parse_int_at(input, &pos);

    /* Process each transaction */
    int tx;
    for (tx = 0; tx < n_tx; tx++) {
        int sender = parse_int_at(input, &pos);
        int receiver = parse_int_at(input, &pos);
        int amount = parse_int_at(input, &pos);

        /* Validate */
        if (amount <= 0) {
            print_str("INVALID tx_");
            print_int(tx);
            print_str("_bad_amount");
            putchar(10);
            return;
        }
        if (sender < 0 || sender >= n_accts || receiver < 0 || receiver >= n_accts) {
            print_str("INVALID tx_");
            print_int(tx);
            print_str("_bad_account");
            putchar(10);
            return;
        }
        if (state[sender] < amount) {
            print_str("INVALID tx_");
            print_int(tx);
            print_str("_insufficient");
            putchar(10);
            return;
        }

        /* Execute */
        state[sender] -= amount;
        state[receiver] += amount;

        /* Check no negative */
        if (state[sender] < 0 || state[receiver] < 0) {
            print_str("INVALID tx_");
            print_int(tx);
            print_str("_negative");
            putchar(10);
            return;
        }
    }

    /* Parse claimed final balances */
    for (i = 0; i < n_accts; i++) {
        final_state[i] = parse_int_at(input, &pos);
    }

    /* Verify final state matches */
    int final_sum = 0;
    for (i = 0; i < n_accts; i++) {
        if (state[i] != final_state[i]) {
            print_str("INVALID state_mismatch_account_");
            print_int(i);
            putchar(10);
            return;
        }
        final_sum += state[i];
    }

    /* Verify supply conservation */
    if (final_sum != supply) {
        print_str("INVALID supply_not_conserved");
        putchar(10);
        return;
    }

    print_str("VALID txs=");
    print_int(n_tx);
    print_str(" supply=");
    print_int(supply);
    putchar(10);
}
