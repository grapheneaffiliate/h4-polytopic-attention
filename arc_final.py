import json
from collections import Counter, deque

BASE = 'C:/Users/atchi/h4-polytopic-attention/data/arc1'
OUTPUT = 'C:/Users/atchi/h4-polytopic-attention/data/arc_python_solutions_b11.json'

# ============================================================
# Task 7468f01a: Extract non-zero rectangle, reverse each row
# ============================================================
def solve_7468f01a(grid):
    rows, cols = len(grid), len(grid[0])
    min_r, max_r, min_c, max_c = rows, 0, cols, 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    result = []
    for r in range(min_r, max_r + 1):
        row = grid[r][min_c:max_c + 1]
        result.append(row[::-1])
    return result

# ============================================================
# Task 746b3537: Deduplicate consecutive identical rows, then columns
# ============================================================
def solve_746b3537(grid):
    deduped = [grid[0]]
    for r in range(1, len(grid)):
        if grid[r] != grid[r - 1]:
            deduped.append(grid[r])
    if not deduped:
        return deduped
    ncols = len(deduped[0])
    keep = [0]
    for c in range(1, ncols):
        col_curr = [deduped[r][c] for r in range(len(deduped))]
        col_prev = [deduped[r][c - 1] for r in range(len(deduped))]
        if col_curr != col_prev:
            keep.append(c)
    return [[row[c] for c in keep] for row in deduped]

# ============================================================
# Task 75b8110e: 8x8 grid, 4 quadrants (TL=4, TR=5, BL=6, BR=9)
# Priority: TR > BL > BR > TL
# ============================================================
def solve_75b8110e(grid):
    H, W = len(grid), len(grid[0])
    h, w = H // 2, W // 2
    TL = [[grid[r][c] for c in range(w)] for r in range(h)]
    TR = [[grid[r][c] for c in range(w, W)] for r in range(h)]
    BL = [[grid[r][c] for c in range(w)] for r in range(h, H)]
    BR = [[grid[r][c] for c in range(w, W)] for r in range(h, H)]
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if TR[r][c] != 0:
                result[r][c] = TR[r][c]
            elif BL[r][c] != 0:
                result[r][c] = BL[r][c]
            elif BR[r][c] != 0:
                result[r][c] = BR[r][c]
            elif TL[r][c] != 0:
                result[r][c] = TL[r][c]
    return result

# ============================================================
# Task 760b3cac: Arrow shape (4s) points direction, 8-pattern reflected that way
# ============================================================
def solve_760b3cac(grid):
    result = [row[:] for row in grid]
    h = len(grid)
    eight_rows = range(0, h // 2)
    arrow_rows = range(h // 2, h)

    bar_row = None
    bar_center = None
    for r in arrow_rows:
        fours = [c for c in range(len(grid[0])) if grid[r][c] == 4]
        if len(fours) == 3:
            bar_row = r
            bar_center = fours[1]
            break

    head_col = None
    for r in arrow_rows:
        if r == bar_row:
            continue
        fours = [c for c in range(len(grid[0])) if grid[r][c] == 4]
        if len(fours) == 1 and fours[0] != bar_center:
            head_col = fours[0]
            break

    if head_col is not None:
        direction = 'left' if head_col < bar_center else 'right'
    else:
        direction = 'left'

    col_start = bar_center - 1
    col_end = bar_center + 2

    for r in eight_rows:
        pattern = grid[r][col_start:col_end]
        reversed_pattern = pattern[::-1]
        if direction == 'left':
            dest_start = col_start - 3
            for i, v in enumerate(reversed_pattern):
                if v != 0:
                    result[r][dest_start + i] = v
        else:
            dest_start = col_end
            for i, v in enumerate(reversed_pattern):
                if v != 0:
                    result[r][dest_start + i] = v
    return result

# ============================================================
# Task 77fdfe62: Frame of 1s with corner colors, 8s map to quadrant colors
# ============================================================
def solve_77fdfe62(grid):
    H, W = len(grid), len(grid[0])
    border_rows = [r for r in range(H) if all(v == 1 for v in grid[r])]
    top_border = border_rows[0]
    bot_border = border_rows[-1]

    left_col = None
    right_col = None
    for c in range(W):
        if all(grid[r][c] == 1 for r in range(top_border, bot_border + 1)):
            if left_col is None:
                left_col = c
            right_col = c

    top_row = grid[top_border - 1]
    bot_row = grid[bot_border + 1]
    tl_color = tr_color = bl_color = br_color = 0

    for c in range(W):
        if top_row[c] not in (0, 1):
            if c <= left_col:
                tl_color = top_row[c]
            else:
                tr_color = top_row[c]
    for c in range(W):
        if bot_row[c] not in (0, 1):
            if c <= left_col:
                bl_color = bot_row[c]
            else:
                br_color = bot_row[c]

    inner_top = top_border + 1
    inner_bot = bot_border - 1
    inner_left = left_col + 1
    inner_right = right_col - 1
    inner_h = inner_bot - inner_top + 1
    inner_w = inner_right - inner_left + 1
    mid_r = inner_h / 2.0
    mid_c = inner_w / 2.0

    result = []
    for r in range(inner_h):
        row = []
        for c in range(inner_w):
            val = grid[inner_top + r][inner_left + c]
            if val == 8:
                if r < mid_r and c < mid_c:
                    row.append(tl_color)
                elif r < mid_r and c >= mid_c:
                    row.append(tr_color)
                elif r >= mid_r and c < mid_c:
                    row.append(bl_color)
                else:
                    row.append(br_color)
            else:
                row.append(0)
        result.append(row)
    return result

# ============================================================
# Task 794b24be: Count 1s, fill positions with 2s in specific order
# ============================================================
def solve_794b24be(grid):
    count = sum(grid[r][c] for r in range(3) for c in range(3))
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    fill_order = [(0, 0), (0, 1), (0, 2), (1, 1)]
    for i in range(min(count, len(fill_order))):
        r, c = fill_order[i]
        result[r][c] = 2
    return result

# ============================================================
# Task 7b7f7511: Find smallest tile that tiles the grid
# ============================================================
def solve_7b7f7511(grid):
    H, W = len(grid), len(grid[0])
    for th in range(1, H + 1):
        if H % th != 0:
            continue
        for tw in range(1, W + 1):
            if W % tw != 0:
                continue
            tile = [grid[r][:tw] for r in range(th)]
            valid = True
            for r in range(H):
                for c in range(W):
                    if grid[r][c] != tile[r % th][c % tw]:
                        valid = False
                        break
                if not valid:
                    break
            if valid and (th < H or tw < W):
                return tile
    return grid

# ============================================================
# Task 7c008303: Row/col of 8s divides grid. Key quadrant colors pattern quadrant.
# ============================================================
def solve_7c008303(grid):
    H, W = len(grid), len(grid[0])

    eight_row = None
    for r in range(H):
        if all(v == 8 for v in grid[r]):
            eight_row = r
            break

    eight_col = None
    for c in range(W):
        if all(grid[r][c] == 8 for r in range(H)):
            eight_col = c
            break

    regions = [
        (0, eight_row, 0, eight_col),
        (0, eight_row, eight_col + 1, W),
        (eight_row + 1, H, 0, eight_col),
        (eight_row + 1, H, eight_col + 1, W),
    ]

    key_data = None
    pattern_data = None

    for r1, r2, c1, c2 in regions:
        qh, qw = r2 - r1, c2 - c1
        q = [[grid[r][c] for c in range(c1, c2)] for r in range(r1, r2)]
        has_content = any(q[r][c] != 0 for r in range(qh) for c in range(qw))
        if not has_content:
            continue
        if key_data is None:
            key_data = q
        else:
            pattern_data = q

    key_h, key_w = len(key_data), len(key_data[0])
    pat_h, pat_w = len(pattern_data), len(pattern_data[0])

    if key_h * key_w > pat_h * pat_w:
        key_data, pattern_data = pattern_data, key_data
        key_h, key_w = len(key_data), len(key_data[0])
        pat_h, pat_w = len(pattern_data), len(pattern_data[0])

    tile_h = pat_h // key_h
    tile_w = pat_w // key_w

    pat_color = 0
    for r in range(pat_h):
        for c in range(pat_w):
            if pattern_data[r][c] != 0:
                pat_color = pattern_data[r][c]
                break
        if pat_color:
            break

    result = [[0] * pat_w for _ in range(pat_h)]
    for kr in range(key_h):
        for kc in range(key_w):
            color = key_data[kr][kc]
            for tr in range(tile_h):
                for tc in range(tile_w):
                    pr = kr * tile_h + tr
                    pc = kc * tile_w + tc
                    if pattern_data[pr][pc] == pat_color:
                        result[pr][pc] = color
    return result

# ============================================================
# Task 7ddcd7ec: 2x2 body with appendages, extend trail in direction
# ============================================================
def solve_7ddcd7ec(grid):
    H, W = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    color = 0
    cells = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0:
                color = grid[r][c]
                cells.add((r, c))

    if not cells:
        return result

    body = set()
    for r, c in cells:
        if (r+1, c) in cells and (r, c+1) in cells and (r+1, c+1) in cells:
            body = {(r, c), (r+1, c), (r, c+1), (r+1, c+1)}
            break

    appendages = cells - body

    for ar, ac in appendages:
        min_dist = float('inf')
        nearest = None
        for br, bc in body:
            d = abs(ar - br) + abs(ac - bc)
            if d < min_dist:
                min_dist = d
                nearest = (br, bc)

        dr = ar - nearest[0]
        dc = ac - nearest[1]

        cr, cc = ar + dr, ac + dc
        while 0 <= cr < H and 0 <= cc < W:
            result[cr][cc] = color
            cr += dr
            cc += dc

    return result

# ============================================================
# Task 7e0986d6: Fix noise cells (replace if >=2 neighbors, else remove)
# ============================================================
def solve_7e0986d6(grid):
    H, W = len(grid), len(grid[0])
    counts = Counter()
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0:
                counts[grid[r][c]] += 1
    main_color = counts.most_common(1)[0][0]
    noise_color = counts.most_common(2)[1][0] if len(counts) > 1 else None

    result = [row[:] for row in grid]
    if noise_color:
        for r in range(H):
            for c in range(W):
                if result[r][c] == noise_color:
                    nn = 0
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != 0:
                            nn += 1
                    if nn >= 2:
                        result[r][c] = main_color
                    else:
                        result[r][c] = 0
    return result


# ============================================================
# Test ALL and save results
# ============================================================
solvers = {
    '7468f01a': solve_7468f01a,
    '746b3537': solve_746b3537,
    '75b8110e': solve_75b8110e,
    '760b3cac': solve_760b3cac,
    '77fdfe62': solve_77fdfe62,
    '794b24be': solve_794b24be,
    '7b7f7511': solve_7b7f7511,
    '7c008303': solve_7c008303,
    '7ddcd7ec': solve_7ddcd7ec,
    '7e0986d6': solve_7e0986d6,
}

output_data = {}
all_pass = True
total_pairs = 0
passed_pairs = 0

for task_id, solver in solvers.items():
    data = json.load(open(f'{BASE}/{task_id}.json'))
    task_pass = True
    task_results = {'train': [], 'test': []}

    for split in ['train', 'test']:
        for i, pair in enumerate(data[split]):
            total_pairs += 1
            inp = pair['input']
            expected = pair['output']
            try:
                got = solver(inp)
            except Exception as e:
                print(f"FAIL {task_id} {split}[{i}]: Exception: {e}")
                import traceback; traceback.print_exc()
                task_pass = False
                task_results[split].append({'pass': False, 'error': str(e)})
                continue

            is_pass = got == expected
            if is_pass:
                passed_pairs += 1
                print(f"PASS {task_id} {split}[{i}]")
            else:
                print(f"FAIL {task_id} {split}[{i}]")
                print(f"  Expected size: {len(expected)}x{len(expected[0])}")
                print(f"  Got size:      {len(got)}x{len(got[0]) if got else 0}")
                task_pass = False

            task_results[split].append({
                'pass': is_pass,
                'output': got
            })

    if not task_pass:
        all_pass = False

    # Store solver code and test outputs
    import inspect
    output_data[task_id] = {
        'solver': inspect.getsource(solver),
        'results': task_results,
        'all_pass': task_pass
    }

# Save to JSON
save_data = {}
for task_id, solver in solvers.items():
    data = json.load(open(f'{BASE}/{task_id}.json'))
    test_outputs = []
    for pair in data['test']:
        test_outputs.append(solver(pair['input']))

    import inspect
    save_data[task_id] = {
        'solver': inspect.getsource(solver),
        'test_output': test_outputs,
        'all_pass': output_data[task_id]['all_pass']
    }

with open(OUTPUT, 'w') as f:
    json.dump(save_data, f, indent=2)

print(f"\n{'='*50}")
print(f"Results: {passed_pairs}/{total_pairs} pairs passed")
print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
print(f"Saved to: {OUTPUT}")
for tid in solvers:
    status = 'PASS' if output_data[tid]['all_pass'] else 'FAIL'
    print(f"  {tid}: {status}")
