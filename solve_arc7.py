import json

def load_task(task_id):
    with open(f'data/arc1/{task_id}.json') as f:
        return json.load(f)

def test_solve(task_id, solve_fn):
    task = load_task(task_id)
    all_pass = True
    for i, pair in enumerate(task['train']):
        out = solve_fn(pair['input'])
        match = out == pair['output']
        print(f"  {task_id} train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
    return all_pass

solutions = {}

# ===================== Task 1190e5a7 =====================
print("=" * 50)
print("Task 1190e5a7")

# The grid is divided by separator lines into cells of different sizes.
# Output is the smallest UNIQUE cell dimensions (not the smallest of each dimension independently).
# Actually, looking at the data:
# Train 0: sep_rows=[2], sep_cols=[1,10,13]
#   Cell heights: [2, 12], Cell widths: [1, 8, 2, 1]
#   Actual cells: 2x1, 2x8, 2x2, 2x1, 12x1, 12x8, 12x2, 12x1
#   Expected output: 2x4 -> this doesn't match any cell!
#
# Wait, maybe I miscounted the separator positions. Let me re-examine.

task1190 = load_task('1190e5a7')

for ti, pair in enumerate(task1190['train']):
    inp = pair['input']
    rows = len(inp)
    cols = len(inp[0])
    out = pair['output']
    out_h = len(out)
    out_w = len(out[0])
    print(f"\n  Train {ti}: {rows}x{cols} -> {out_h}x{out_w}")

    # Let me look at the actual grid values
    # Find separator value
    sep = None
    for r in range(rows):
        if len(set(inp[r])) == 1 and inp[r][0] != inp[0][0]:
            sep = inp[r][0]
            break
    if sep is None:
        # The first row might itself be a separator
        for r in range(rows):
            if len(set(inp[r])) == 1:
                sep = inp[r][0]
                break

    bg = None
    for r in range(rows):
        for c in range(cols):
            if inp[r][c] != sep:
                bg = inp[r][c]
                break
        if bg is not None:
            break

    print(f"    sep={sep}, bg={bg}")

    # Find separator rows and columns
    sep_rows = []
    for r in range(rows):
        if all(inp[r][c] == sep for c in range(cols)):
            sep_rows.append(r)

    sep_cols = []
    for c in range(cols):
        if all(inp[r][c] == sep for r in range(rows)):
            sep_cols.append(c)

    print(f"    sep_rows={sep_rows}")
    print(f"    sep_cols={sep_cols}")

    # Actually, separator rows/cols might be GROUPS of consecutive rows/cols
    # Let me look at which rows are separators
    # And compute the gaps between them (which are the cell rows)
    all_rows = sorted(set([-1] + sep_rows + [rows]))
    all_cols = sorted(set([-1] + sep_cols + [cols]))

    cell_heights = []
    for i in range(len(all_rows)-1):
        gap = all_rows[i+1] - all_rows[i] - 1
        if gap > 0:
            cell_heights.append(gap)

    cell_widths = []
    for i in range(len(all_cols)-1):
        gap = all_cols[i+1] - all_cols[i] - 1
        if gap > 0:
            cell_widths.append(gap)

    print(f"    cell_heights={cell_heights}")
    print(f"    cell_widths={cell_widths}")

    # Maybe the output is the MOST COMMON cell height x MOST COMMON width?
    from collections import Counter
    h_counts = Counter(cell_heights)
    w_counts = Counter(cell_widths)
    print(f"    h_counts={h_counts}")
    print(f"    w_counts={w_counts}")

    # Or maybe the output is the UNIQUE cell that appears only once?
    # Train 0 expected 2x4. Heights are [2, 12], widths are [1, 8, 2, 1].
    # No 4 in widths!

    # Wait, maybe I'm misidentifying separators. Let me check:
    # Are there separator rows that aren't full-width? Or columns that aren't full-height?
    # Maybe the separator pattern is more complex.

    # Let me print the actual grid
    if rows <= 20:
        print("    Grid:")
        for r in range(rows):
            print(f"      {''.join(str(x) for x in inp[r])}")

# Actually let me look at the grid values for train 0:
pair0 = task1190['train'][0]
inp0 = pair0['input']
print("\n\nTrain 0 grid analysis:")
rows0, cols0 = len(inp0), len(inp0[0])
# Check rows
for r in range(rows0):
    is_sep = all(inp0[r][c] == 7 for c in range(cols0))
    print(f"  Row {r}: {'SEP' if is_sep else '   '} {inp0[r]}")
# Check cols
print("\n  Column analysis:")
for c in range(cols0):
    is_sep = all(inp0[r][c] == 7 for r in range(rows0))
    col_vals = [inp0[r][c] for r in range(rows0)]
    if is_sep:
        print(f"  Col {c}: SEP")

# Train 1
pair1 = task1190['train'][1]
inp1 = pair1['input']
print(f"\nTrain 1 grid ({len(inp1)}x{len(inp1[0])}):")
for r in range(len(inp1)):
    is_sep = all(inp1[r][c] == 8 for c in range(len(inp1[0])))
    print(f"  Row {r}: {'SEP' if is_sep else '   '} {inp1[r]}")
print("  Sep cols:")
for c in range(len(inp1[0])):
    is_sep = all(inp1[r][c] == 8 for r in range(len(inp1)))
    if is_sep:
        print(f"    Col {c}: SEP")
