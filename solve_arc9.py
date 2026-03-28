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

# ===================== Task 09629e4f =====================
task09 = load_task('09629e4f')

# Let me try looking at this from a GLOBAL perspective.
# For each color, extract the 9 positions (one per block) as a 3x3 grid of (cell_r, cell_c).
# Then see which color has its positions forming a specific pattern.

# For each color, check if the cell_row values in each block-row are all the same,
# or if cell_col values in each block-col are all the same.

# Actually, let me try: maybe the output for block (br, bc) is determined by which
# color has cell_row == br IN THAT SPECIFIC BLOCK. Checked before, but let me
# look more carefully at which blocks have exactly ONE color with cell_row==br.

for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"\nTrain {ti}:")

    for br in range(3):
        for bc in range(3):
            rs, cs = br*4, bc*4
            out_val = out[rs][cs]
            # Find colors where cell_row == br
            row_matches = []
            col_matches = []
            both_matches = []
            for r in range(3):
                for c in range(3):
                    v = inp[rs+r][cs+c]
                    if v in [0, 5]:
                        continue
                    if r == br and c == bc:
                        both_matches.append(v)
                    elif r == br:
                        row_matches.append(v)
                    elif c == bc:
                        col_matches.append(v)

            # Color where cell_row == bc and cell_col == br (transposed)
            trans_matches = []
            for r in range(3):
                for c in range(3):
                    v = inp[rs+r][cs+c]
                    if v in [0, 5]:
                        continue
                    if r == bc and c == br:
                        trans_matches.append(v)

            if out_val != 0:
                print(f"  Block ({br},{bc}) out={out_val}: at(br,bc)={both_matches}, at(bc,br)={trans_matches}, row_br={row_matches+both_matches}, col_bc={col_matches+both_matches}")

# Maybe the output is determined by a lookup table that maps the 3x3 block pattern
# to a color. Let me check if the zero positions uniquely determine the block.

# Actually, I think the answer might be about PAIR matching between blocks.
# Each block is described by the arrangement of 5 colors (or 4 colors + empty position).
# The output identifies which blocks "match" some criterion.

# Let me try a completely new approach: brute force.
# For each block, compute some hash/feature, and see which feature correlates with non-zero output.

# Feature 1: position of 0 cells
# Feature 2: which cell position has which color
# Feature 3: sum of color values
# Feature 4: color at specific positions

print("\n\n=== Trying: output = color where cell_row + cell_col == br + bc (mod 3) ===")
for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"\nTrain {ti}:")

    for br in range(3):
        for bc in range(3):
            rs, cs = br*4, bc*4
            out_val = out[rs][cs]
            target_sum = (br + bc) % 3
            matches = []
            for r in range(3):
                for c in range(3):
                    v = inp[rs+r][cs+c]
                    if v in [0, 5, 8]:
                        continue
                    if (r + c) % 3 == target_sum:
                        matches.append(v)
            if out_val != 0:
                print(f"  Block ({br},{bc}) out={out_val}: matches for sum={(br+bc)%3} -> {matches}")

# Let me try modular arithmetic on the cell position
print("\n\n=== Trying various cell position formulas ===")
# For the output blocks, what formula of (br, bc) gives (r, c)?
for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    print(f"\nTrain {ti}:")

    for br in range(3):
        for bc in range(3):
            rs, cs = br*4, bc*4
            out_val = out[rs][cs]
            if out_val == 0 or out_val == 5:
                continue
            # Find where out_val is in this block
            for r in range(3):
                for c in range(3):
                    if inp[rs+r][cs+c] == out_val:
                        # What's the relationship between (br,bc) and (r,c)?
                        print(f"  Block ({br},{bc}) out={out_val} at cell ({r},{c}): "
                              f"r-br={r-br}, c-bc={c-bc}, "
                              f"(r+br)%3={(r+br)%3}, (c+bc)%3={(c+bc)%3}, "
                              f"(r*br)%3={(r*br)%3 if br else 'N/A'}")

# Interesting patterns:
# Let me check if r+br is constant and c+bc is constant for the output color.
print("\n\n=== Check (r+br)%3 and (c+bc)%3 for output colors ===")
for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']
    rbc_vals = set()
    cbc_vals = set()

    for br in range(3):
        for bc in range(3):
            rs, cs = br*4, bc*4
            out_val = out[rs][cs]
            if out_val == 0 or out_val == 5:
                continue
            for r in range(3):
                for c in range(3):
                    if inp[rs+r][cs+c] == out_val:
                        rbc_vals.add((r+br)%3)
                        cbc_vals.add((c+bc)%3)
    print(f"  Train {ti}: (r+br)%3 = {rbc_vals}, (c+bc)%3 = {cbc_vals}")
