/*
 * SOLVER: Sort an integer array (insertion sort)
 * Input: "N a1 a2 ... aN"
 * Output: sorted elements, space-separated
 */
__attribute__((noinline, optnone))
void compute(const char *input) {
    int arr[64];
    int n = 0;
    int pos = 0;

    /* Parse: first number is count, rest are elements */
    int val = 0;
    int neg = 0;
    int has_val = 0;
    int i;

    for (i = 0; input[i] != 0; i++) {
        if (input[i] >= '0' && input[i] <= '9') {
            val = val * 10 + (input[i] - '0');
            has_val = 1;
        } else if (input[i] == '-') {
            neg = 1;
        } else if (has_val) {
            if (neg) val = -val;
            arr[n] = val;
            n++;
            val = 0;
            neg = 0;
            has_val = 0;
        }
    }
    if (has_val) {
        if (neg) val = -val;
        arr[n] = val;
        n++;
    }

    /* Insertion sort */
    for (i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }

    /* Output */
    for (i = 0; i < n; i++) {
        if (i > 0) putchar(32);
        print_int(arr[i]);
    }
    putchar(10);
}
