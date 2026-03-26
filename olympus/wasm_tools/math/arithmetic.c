/* Exact integer arithmetic via transformer-vm.
 *
 * Input format: "a op b"
 *   op: + - * / %
 *   Examples: "15 + 23", "99 * 99", "1000 / 7", "17 % 5"
 *
 * Output: decimal result followed by newline.
 * All computation is exact (i32).
 */

void compute(const char *input) {
    /* Parse first number (possibly negative) */
    int pos = 0;
    int neg_a = 0;
    if (input[pos] == '-') { neg_a = 1; pos++; }

    int a = 0;
    while (input[pos] >= '0' && input[pos] <= '9') {
        int d = input[pos] - '0';
        int t2 = a + a;
        int t4 = t2 + t2;
        int t8 = t4 + t4;
        a = t8 + t2 + d;  /* a = a * 10 + d */
        pos++;
    }
    if (neg_a) a = 0 - a;

    /* Skip spaces */
    while (input[pos] == ' ') pos++;

    /* Parse operator */
    int op = input[pos];
    pos++;

    /* Skip spaces */
    while (input[pos] == ' ') pos++;

    /* Parse second number (possibly negative) */
    int neg_b = 0;
    if (input[pos] == '-') { neg_b = 1; pos++; }

    int b = 0;
    while (input[pos] >= '0' && input[pos] <= '9') {
        int d = input[pos] - '0';
        int t2 = b + b;
        int t4 = t2 + t2;
        int t8 = t4 + t4;
        b = t8 + t2 + d;
        pos++;
    }
    if (neg_b) b = 0 - b;

    /* Compute */
    int result = 0;
    if (op == '+')      result = a + b;
    else if (op == '-') result = a - b;
    else if (op == '*') result = a * b;
    else if (op == '/') {
        if (b != 0) result = a / b;
    }
    else if (op == '%') {
        if (b != 0) result = a % b;
    }
    else if (op == '^') {
        /* Power: a^b for non-negative b */
        result = 1;
        int i;
        int exp = b;
        if (exp < 0) exp = 0;
        for (i = 0; i < exp; i++) {
            result = result * a;
        }
    }

    /* Output */
    printf("%d\n", result);
}
