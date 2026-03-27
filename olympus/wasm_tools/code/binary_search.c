/* Binary search in sorted array.
 * Input: "target val1 val2 val3 ..." (first number is target)
 * Output: index (0-based) or -1 if not found
 */
void compute(const char *input) {
    int arr[256];
    int n = 0;
    int pos = 0;
    /* Parse all numbers */
    while (input[pos]) {
        while (input[pos] == ' ') pos++;
        if (input[pos] == 0) break;
        int neg = 0;
        if (input[pos] == '-') { neg = 1; pos++; }
        int val = 0;
        while (input[pos] >= '0' && input[pos] <= '9') {
            int d = input[pos] - '0';
            int t2 = val + val; int t4 = t2 + t2; int t8 = t4 + t4;
            val = t8 + t2 + d;
            pos++;
        }
        if (neg) val = 0 - val;
        arr[n] = val;
        n++;
    }
    if (n < 2) { print_int(0 - 1); putchar('\n'); return; }
    int target = arr[0];
    int lo = 1, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] == target) {
            print_int(mid - 1);
            putchar('\n');
            return;
        }
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    print_int(0 - 1);
    putchar('\n');
}
