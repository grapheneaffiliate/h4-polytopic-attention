import json

def load_task(task_id):
    with open(f'data/arc1/{task_id}.json') as f:
        return json.load(f)

task09 = load_task('09629e4f')

# Let me try yet another approach. For each pair of blocks that share a row or column,
# look at what colors appear at the "boundary" positions (edges facing each other).
# Or: maybe the grid represents a permutation matrix and each block encodes a permutation.

# New idea: maybe each block can be decoded as a number, and the output is determined
# by which block in each row/column has a specific property.

# Actually, the simplest thing I haven't tried: maybe the output color for block (br,bc)
# equals the color at position (zero_row, zero_col) where (zero_row, zero_col) is
# one of the 4-5 zero positions in the block.

# Wait, I just realized something. Each block has exactly 4 or 5 non-zero cells.
# The block missing 8 has 4 non-zero cells and 5 zeros.
# Other blocks have 5 non-zero cells and 4 zeros.
#
# What if: the non-zero cells in each block define a PERMUTATION or a function
# from positions to colors? And the output is determined by evaluating the function
# at a specific position.

# Let me look at this from the perspective of the FULL 9x9 grid (excluding separators).
# The 9x9 grid has 81 cells. Each non-zero, non-separator cell has a value in {0,2,3,4,6,8}.
#
# For each training pair, let me construct the full 9x9 grid and look for global patterns.

for ti in range(1):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']

    # Extract the 9x9 inner grid (removing separator rows/cols)
    grid9 = []
    for br in range(3):
        for r in range(3):
            row = []
            for bc in range(3):
                for c in range(3):
                    row.append(inp[br*4 + r][bc*4 + c])
            grid9.append(row)

    print(f"\nTrain {ti} 9x9 grid:")
    for r in range(9):
        print(f"  {grid9[r]}")

    # For each color, find ALL positions in the 9x9 grid
    print(f"\n  Color positions in 9x9:")
    for color in [2, 3, 4, 6, 8]:
        positions = [(r, c) for r in range(9) for c in range(9) if grid9[r][c] == color]
        print(f"    Color {color}: {positions}")
        # Check if positions form a specific pattern
        # Do they form a permutation matrix (one per row, one per col)?
        rows = set(r for r, c in positions)
        cols = set(c for r, c in positions)
        print(f"      Rows: {sorted(rows)}, Cols: {sorted(cols)}")
        # Is it one per row? one per col?
        row_counts = {}
        col_counts = {}
        for r, c in positions:
            row_counts[r] = row_counts.get(r, 0) + 1
            col_counts[c] = col_counts.get(c, 0) + 1
        print(f"      Row counts: {dict(sorted(row_counts.items()))}")
        print(f"      Col counts: {dict(sorted(col_counts.items()))}")

# Check if each color appears exactly once per row in the 9x9 grid
# (Like a Sudoku where each color appears once per row and once per column)

print("\n\n=== Checking if colors form permutation matrices in 9x9 grid ===")
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
    for color in [2, 3, 4, 6, 8]:
        positions = [(r, c) for r in range(9) for c in range(9) if grid9[r][c] == color]
        # Check: exactly one per row?
        rows = [r for r, c in positions]
        cols = [c for r, c in positions]
        one_per_row = (len(set(rows)) == len(rows) == 9)
        one_per_col = (len(set(cols)) == len(cols) == 9)
        # Check: exactly one per 3x3 block?
        blocks = [(r//3, c//3) for r, c in positions]
        one_per_block = (len(set(blocks)) == len(blocks) == 9)
        print(f"  Color {color}: count={len(positions)}, one_per_row={one_per_row}, one_per_col={one_per_col}, one_per_block={one_per_block}")

    out_grid = [[out[br*4][bc*4] for bc in range(3)] for br in range(3)]
    print(f"  Output: {out_grid}")
