/*
 * TVM-VERIFIED LEDGER: Transfer transaction
 *
 * Executes a transfer and verifies all invariants:
 *   1. Sender has sufficient balance
 *   2. No negative balances after transfer
 *   3. Total money supply is conserved (sum before = sum after)
 *   4. Amount is positive
 *
 * Input: "sender_bal receiver_bal amount"
 * Output: "VALID new_sender_bal new_receiver_bal" or "INVALID reason"
 *
 * This replaces: banks, clearinghouses, auditors, and 2-day settlement.
 * The proof IS the settlement. Mathematical certainty, not institutional trust.
 */

__attribute__((noinline, optnone))
void compute(const char *input) {
    int sender_bal, receiver_bal, amount;

    /* Parse three integers */
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
    sender_bal = vals[0];
    receiver_bal = vals[1];
    amount = vals[2];

    /* Invariant 1: amount must be positive */
    if (amount <= 0) {
        print_str("INVALID amount_not_positive");
        putchar(10);
        return;
    }

    /* Invariant 2: sender has sufficient balance */
    if (sender_bal < amount) {
        print_str("INVALID insufficient_balance");
        putchar(10);
        return;
    }

    /* Execute transfer */
    int new_sender = sender_bal - amount;
    int new_receiver = receiver_bal + amount;

    /* Invariant 3: no negative balances */
    if (new_sender < 0 || new_receiver < 0) {
        print_str("INVALID negative_balance");
        putchar(10);
        return;
    }

    /* Invariant 4: conservation of money supply */
    int supply_before = sender_bal + receiver_bal;
    int supply_after = new_sender + new_receiver;
    if (supply_before != supply_after) {
        print_str("INVALID supply_not_conserved");
        putchar(10);
        return;
    }

    /* Invariant 5: overflow check */
    if (new_receiver < receiver_bal) {
        print_str("INVALID overflow");
        putchar(10);
        return;
    }

    /* All invariants hold — transaction is valid */
    print_str("VALID ");
    print_int(new_sender);
    putchar(32);
    print_int(new_receiver);
    putchar(10);
}
