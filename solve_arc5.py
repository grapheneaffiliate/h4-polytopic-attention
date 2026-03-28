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
# Maybe each color has a "home position" within the 3x3 cell grid,
# and the block where color X is at its home position gets output X.
# The home position for each color could be determined by looking at the
# block that's missing color 8 (it only has 4 colors, not 5).

# In the block missing 8, the 4 colors are at specific positions.
# Those positions could define the "home" for each color.

task09 = load_task('09629e4f')

for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']

    # Find the block missing 8
    missing_block = None
    for br in range(3):
        for bc in range(3):
            rs, cs = br*4, bc*4
            has8 = False
            for r in range(3):
                for c in range(3):
                    if inp[rs+r][cs+c] == 8:
                        has8 = True
            if not has8:
                missing_block = (br, bc)

    print(f"\nTrain {ti}: block missing 8 = {missing_block}")

    # Get the color->position mapping from this block
    br, bc = missing_block
    rs, cs = br*4, bc*4
    home = {}  # color -> (cell_r, cell_c)
    for r in range(3):
        for c in range(3):
            v = inp[rs+r][cs+c]
            if v not in [0, 5]:
                home[v] = (r, c)
    print(f"  Home positions: {home}")

    # Now for each other block, check which color is at its home position
    print(f"  Checking all blocks:")
    for br2 in range(3):
        for bc2 in range(3):
            rs2, cs2 = br2*4, bc2*4
            out_val = out[rs2][cs2]
            matched_color = None
            for r in range(3):
                for c in range(3):
                    v = inp[rs2+r][cs2+c]
                    if v in home and home[v] == (r, c):
                        matched_color = v
            print(f"    Block ({br2},{bc2}): matched={matched_color}, expected={out_val}")

# Let me try a different "home" mapping. Maybe each color's home is determined
# by where 8 appears in each block (8 occupies the position that the output
# color "should" be at).

print("\n\n=== Alternative: 8 position indicates which color should fill the block ===")
for ti in range(4):
    pair = task09['train'][ti]
    inp = pair['input']
    out = pair['output']

    # Find block missing 8 to get position->color mapping
    key_block = None
    for br in range(3):
        for bc in range(3):
            rs, cs = br*4, bc*4
            has8 = any(inp[rs+r][cs+c] == 8 for r in range(3) for c in range(3))
            if not has8:
                key_block = (br, bc)

    br, bc = key_block
    rs, cs = br*4, bc*4
    pos_to_color = {}
    for r in range(3):
        for c in range(3):
            v = inp[rs+r][cs+c]
            if v not in [0, 5]:
                pos_to_color[(r, c)] = v

    print(f"\nTrain {ti}: key block = {key_block}, pos_to_color = {pos_to_color}")

    # For each block with 8, find 8's position, look up which color maps to that position
    for br2 in range(3):
        for bc2 in range(3):
            rs2, cs2 = br2*4, bc2*4
            out_val = out[rs2][cs2]
            pos8 = None
            for r in range(3):
                for c in range(3):
                    if inp[rs2+r][cs2+c] == 8:
                        pos8 = (r, c)
            if pos8 and pos8 in pos_to_color:
                predicted = pos_to_color[pos8]
            else:
                predicted = 0  # no 8, so block itself is the key
            print(f"    Block ({br2},{bc2}): pos8={pos8}, predicted={predicted}, expected={out_val}")
