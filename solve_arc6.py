import json

def load_task(task_id):
    with open(f'data/arc1/{task_id}.json') as f:
        return json.load(f)

task09 = load_task('09629e4f')

# Let me look at the cell positions of each color across ALL blocks as a 9-element sequence.
# For each color, in each block, the color occupies one cell position (r,c) within the block.
# This gives us a 3x3 grid of positions for each color.

for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"\nTrain {ti}:")

    for color in [2, 3, 4, 6, 8]:
        pos_grid = [[None]*3 for _ in range(3)]
        for br in range(3):
            for bc in range(3):
                rs, cs = br*4, bc*4
                for r in range(3):
                    for c in range(3):
                        if inp[rs+r][cs+c] == color:
                            pos_grid[br][bc] = (r, c)
        # Print positions as a grid
        print(f"  Color {color}: ", end="")
        for br in range(3):
            for bc in range(3):
                p = pos_grid[br][bc]
                if p:
                    print(f"({p[0]},{p[1]})", end=" ")
                else:
                    print("None  ", end=" ")
            if br < 2:
                print("| ", end="")
        print()

    # Output
    out_grid = [[out[br*4][bc*4] for bc in range(3)] for br in range(3)]
    print(f"  Output: {out_grid}")

# Now let me look for a pattern. Maybe for each color, the positions in the 3x3
# grid of blocks form a specific pattern (like a Latin square), and the output
# is determined by which color has a special position arrangement.

# Actually, for each color, the 9 cell positions (r,c) within each block should
# form some kind of pattern. Let me look at row and column indices separately.

print("\n\n=== Row indices of each color in each block ===")
for ti in range(1):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"\nTrain {ti}:")

    for color in [2, 3, 4, 6, 8]:
        row_grid = [[None]*3 for _ in range(3)]
        col_grid = [[None]*3 for _ in range(3)]
        for br in range(3):
            for bc in range(3):
                rs, cs = br*4, bc*4
                for r in range(3):
                    for c in range(3):
                        if inp[rs+r][cs+c] == color:
                            row_grid[br][bc] = r
                            col_grid[br][bc] = c
        print(f"  Color {color} rows: {row_grid}")
        print(f"  Color {color} cols: {col_grid}")

    out_grid = [[out[br*4][bc*4] for bc in range(3)] for br in range(3)]
    print(f"  Output: {out_grid}")

# Maybe the answer relates to looking at each color's position as (r,c) within
# the block, and checking if r equals the block_row or c equals block_col.

print("\n\n=== Check if cell_row == block_row or cell_col == block_col ===")
for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"\nTrain {ti}:")

    for br in range(3):
        for bc in range(3):
            rs, cs = br*4, bc*4
            out_val = out[rs][cs]
            matches = {}
            for r in range(3):
                for c in range(3):
                    v = inp[rs+r][cs+c]
                    if v not in [0, 5]:
                        # Check various conditions
                        cond_rr = (r == br)
                        cond_cc = (c == bc)
                        cond_rc = (r == bc)
                        cond_cr = (c == br)
                        matches[v] = {'rr': cond_rr, 'cc': cond_cc, 'rc': cond_rc, 'cr': cond_cr}
            if out_val != 0:
                for v, m in matches.items():
                    if v == out_val:
                        print(f"  Block ({br},{bc}) out={out_val}: conditions = {m}")

# Let me also check: for each block, which color is at position (block_row, block_col)?
print("\n\n=== Color at position (br, bc) in each block ===")
for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"\nTrain {ti}:")

    for br in range(3):
        for bc in range(3):
            rs, cs = br*4, bc*4
            val_at_diag = inp[rs+br][cs+bc]
            out_val = out[rs][cs]
            print(f"  Block ({br},{bc}): val at ({br},{bc}) = {val_at_diag}, output = {out_val}")

# Maybe each color appears at position (br, bc) in exactly one block,
# and that block gets the color?
print("\n\n=== Which block has each color at position (br,bc)? ===")
for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"\nTrain {ti}:")
    out_grid = [[out[br*4][bc*4] for bc in range(3)] for br in range(3)]
    print(f"  Output: {out_grid}")

    for color in [2, 3, 4, 6]:
        found = []
        for br in range(3):
            for bc in range(3):
                rs, cs = br*4, bc*4
                if inp[rs+br][cs+bc] == color:
                    found.append((br, bc))
        print(f"  Color {color} at (br,bc): {found}")
