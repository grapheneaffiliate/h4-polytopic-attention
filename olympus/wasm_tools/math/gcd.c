/* GCD and LCM via Euclidean algorithm.
 * Input: "a b", e.g. "24 36"
 * Output: "gcd=12 lcm=72\n"
 */

void compute(const char *input) {
    /* Parse first number */
    int pos = 0;
    int a = 0;
    while (input[pos] >= '0' && input[pos] <= '9') {
        int d = input[pos] - '0';
        int t2 = a + a;
        int t4 = t2 + t2;
        int t8 = t4 + t4;
        a = t8 + t2 + d;
        pos++;
    }
    /* Skip space */
    while (input[pos] == ' ') pos++;

    /* Parse second number */
    int b = 0;
    while (input[pos] >= '0' && input[pos] <= '9') {
        int d = input[pos] - '0';
        int t2 = b + b;
        int t4 = t2 + t2;
        int t8 = t4 + t4;
        b = t8 + t2 + d;
        pos++;
    }

    int orig_a = a, orig_b = b;

    /* Euclidean algorithm */
    while (b != 0) {
        int r = a % b;
        a = b;
        b = r;
    }
    int gcd = a;

    /* LCM = |orig_a * orig_b| / gcd */
    int lcm = 0;
    if (gcd != 0) {
        lcm = (orig_a / gcd) * orig_b;
    }

    print_str("gcd=");
    print_int(gcd);
    print_str(" lcm=");
    print_int(lcm);
    putchar('\n');
}
