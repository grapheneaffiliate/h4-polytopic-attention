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

task09 = load_task('09629e4f')

# Each color appears once per block. This is like a Sudoku.
# But not one per row/col. Let me check if it's one per 3-row band and one per 3-col band.
# In Sudoku, this would mean each color appears exactly 3 times in rows 0-2, 3 times in 3-5, 3 times in 6-8.

for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']

    grid9 = []
    for br in range(3):
        for r in range(3):
            row = []
            for bc in range(3):
                for c in range(3):
                    row.append(inp[br*4 + r][bc*4 + c])
            grid9.append(row)

    print(f"\nTrain {ti}:")
    # For each color, check distribution across rows and columns
    for color in [2, 3, 4, 6]:
        positions = [(r, c) for r in range(9) for c in range(9) if grid9[r][c] == color]
        # Check per-row distribution
        row_band_counts = [0, 0, 0]
        col_band_counts = [0, 0, 0]
        for r, c in positions:
            row_band_counts[r // 3] += 1
            col_band_counts[c // 3] += 1

        # Check per-row and per-col
        rows = [r for r, c in positions]
        cols = [c for r, c in positions]
        # Is there exactly one per row in any row-band?
        for band in range(3):
            band_rows = [r for r in rows if r // 3 == band]
            band_cols = [c for r, c in positions if r // 3 == band]
            unique_rows = len(set(band_rows)) == len(band_rows) == 3
            unique_cols = len(set(band_cols)) == len(band_cols) == 3
            if unique_rows and unique_cols:
                # This band has a permutation matrix for this color
                pass

        # Check if color forms a valid Sudoku number (one per row, col, block)
        # Actually, the grid is 9x9 but only 5 "numbers" and many zeros.
        # It's more like a partial Sudoku.

    # Let me try a completely different approach.
    # Maybe the answer is: the output at block (br, bc) is the color
    # that would complete a Latin square property.
    # Or: the non-zero output cells form a valid placement of 4 queens or something.

    # Let me just check: is the output always such that each output color appears
    # at a different row and column of the 3x3 output grid?
    out_grid = [[out[br*4][bc*4] for bc in range(3)] for br in range(3)]
    non_zero = [(br, bc, out_grid[br][bc]) for br in range(3) for bc in range(3) if out_grid[br][bc] != 0]
    output_rows = [br for br, bc, v in non_zero]
    output_cols = [bc for br, bc, v in non_zero]
    print(f"  Output non-zero: {non_zero}")
    print(f"  Unique output rows: {len(set(output_rows))}, Unique output cols: {len(set(output_cols))}")
    print(f"  Output values: {sorted(set(v for _, _, v in non_zero))}")

# Each output has 4 non-zero values from {2,3,4,6}, and they always occupy
# exactly 4 of the 9 positions. Some rows/cols have 0, 1, or 2 non-zero values.
# Not a permutation matrix for the outputs.

# Let me try ONE MORE thing: the answer might be about the DIAGONAL.
# For each 3x3 block, the cells on the main diagonal (0,0), (1,1), (2,2) are special.
# Or the anti-diagonal (0,2), (1,1), (2,0).

print("\n\n=== Diagonal analysis ===")
for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"\nTrain {ti}:")
    for br in range(3):
        for bc in range(3):
            rs, cs = br*4, bc*4
            out_val = out[rs][cs]
            main_diag = [inp[rs+i][cs+i] for i in range(3)]
            anti_diag = [inp[rs+i][cs+2-i] for i in range(3)]
            print(f"  Block ({br},{bc}): main_diag={main_diag}, anti_diag={anti_diag}, out={out_val}")

# Hmm, let me also check: is the output related to which color is NOT on the diagonals?

# Actually, I realize I should look at this from the perspective of
# color 0 (empty cells). Each block has 4 empty cells and 5 filled cells
# (or 5 empty and 4 filled for the block missing 8).
# The 4 empty cells form a pattern. Maybe the empty cell pattern
# is the key.

print("\n\n=== Empty cell patterns ===")
for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"\nTrain {ti}:")
    for br in range(3):
        for bc in range(3):
            rs, cs = br*4, bc*4
            out_val = out[rs][cs]
            zeros = []
            non_zeros = {}
            for r in range(3):
                for c in range(3):
                    v = inp[rs+r][cs+c]
                    if v == 0:
                        zeros.append((r, c))
                    else:
                        non_zeros[(r,c)] = v
            # Convert zeros to a tuple for hashing
            z_key = tuple(sorted(zeros))
            print(f"  Block ({br},{bc}): zeros={z_key}, out={out_val}")

# Let me check if blocks with the same zero pattern always have the same output behavior.

print("\n\n=== Grouping blocks by zero pattern ===")
pattern_to_outputs = {}
for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    for br in range(3):
        for bc in range(3):
            rs, cs = br*4, bc*4
            out_val = out[rs][cs]
            zeros = tuple(sorted((r, c) for r in range(3) for c in range(3) if inp[rs+r][cs+c] == 0))
            if zeros not in pattern_to_outputs:
                pattern_to_outputs[zeros] = []
            pattern_to_outputs[zeros].append((ti, br, bc, out_val))

for pattern, entries in sorted(pattern_to_outputs.items()):
    outputs = [e[3] for e in entries]
    if any(o != 0 for o in outputs):
        print(f"  Pattern {pattern}: outputs={outputs}")

# If the same zero pattern always gives the same output (0 or non-zero), that would be useful.

# Actually, let me take a completely different approach. I'll look at the
# relationship between pairs of blocks.
# Maybe the output is determined by: for each block, look at the block at the
# TRANSPOSED position (br,bc) -> (bc,br), and the output color is the color
# that appears at the cell position that corresponds to where the OTHER block's
# special color is.

# Or maybe the simplest thing: within each block, the output color is the one
# NOT appearing at any of the 0-positions. I.e., the output is determined by
# which color FILLS the block such that every non-zero cell matches the output
# or the existing color matches the output somehow.

# Let me try: the output for each block is the non-8 color that,
# when combined with the existing non-zero cells, would complete
# some specific pattern (like filling all cells in a row or column).

# Actually, I wonder if each block represents a cell in a Sudoku,
# and we need to "solve" the Sudoku by finding which value (from {2,3,4,6})
# goes in each cell. The Sudoku constraints are that each value appears
# once per row and once per column of the 3x3 output grid.

# But the output doesn't satisfy this: in train 0, output row 0 = [2,0,0],
# row 1 = [0,4,3], row 2 = [6,0,0]. Column 0 has values 2,0,6 (2 non-zero).
# Column 1 has 0,4,0 (1 non-zero). So it's NOT a Latin square.

# OK I give up trying to analytically find the pattern for 09629e4f.
# Let me move on and save the solutions I have.

# Let me also re-verify all passing solutions
print("\n" + "=" * 50)
print("Verifying all solutions:")

# Load solutions
with open('data/arc_python_solutions.json') as f:
    existing = json.load(f)
print(f"Existing solutions: {list(existing.keys())}")

# Re-test each
for task_id, sol_data in existing.items():
    if isinstance(sol_data, dict):
        code = sol_data['code']
    else:
        code = sol_data
    exec(code, globals())
    task = load_task(task_id)
    all_pass = True
    for i, pair in enumerate(task['train']):
        out = solve(pair['input'])
        if out != pair['output']:
            all_pass = False
            print(f"  {task_id} train {i}: FAIL")
    if all_pass:
        print(f"  {task_id}: ALL PASS")
