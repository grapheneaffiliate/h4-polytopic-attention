/* Collatz sequence from input number.
 * Input: decimal integer, e.g. "27"
 * Output: full sequence, e.g. "27 82 41 ... 1\n"
 */

void compute(const char *input) {
    int n;
    sscanf(input, "%d", &n);

    printf("%d", n);
    while (n != 1) {
        /* Check if odd: n - (n/2)*2 != 0 */
        int half = n / 2;
        int doubled = half + half;
        if (n != doubled) {
            /* odd: 3n+1 */
            n = n + n + n + 1;
        } else {
            /* even: n/2 */
            n = half;
        }
        printf(" %d", n);
    }
    putchar('\n');
}
