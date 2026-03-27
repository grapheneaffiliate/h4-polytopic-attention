/* Sort integers.
 * Input: space-separated integers
 * Output: sorted integers, space-separated
 */
void compute(const char *input) {
    int arr[256];
    int n = 0;
    int pos = 0;
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
    /* Insertion sort */
    int i, j;
    for (i = 1; i < n; i++) {
        int key = arr[i];
        j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
    for (i = 0; i < n; i++) {
        if (i > 0) putchar(' ');
        print_int(arr[i]);
    }
    putchar('\n');
}
