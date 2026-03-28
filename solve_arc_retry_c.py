import json

solutions = {}

solutions["a8d7556c"] = """def solve(grid):
    rows, cols = len(grid), len(grid[0])
    # Find all maximal rectangles of 0s with both dimensions >= 2
    rects = []
    for r1 in range(rows):
        for c1 in range(cols):
            if grid[r1][c1] != 0:
                continue
            c2 = c1
            while c2 + 1 < cols and grid[r1][c2 + 1] == 0:
                c2 += 1
            for ce in range(c1, c2 + 1):
                if ce - c1 + 1 < 2:
                    continue
                r2 = r1
                while r2 + 1 < rows:
                    ok = all(grid[r2 + 1][cc] == 0 for cc in range(c1, ce + 1))
                    if ok:
                        r2 += 1
                    else:
                        break
                if r2 - r1 + 1 >= 2:
                    rects.append((r1, c1, r2, ce))
    rects = list(set(rects))
    maximal = []
    for i, (r1, c1, r2, c2) in enumerate(rects):
        is_max = True
        for j, (r1b, c1b, r2b, c2b) in enumerate(rects):
            if i != j and r1b <= r1 and c1b <= c1 and r2b >= r2 and c2b >= c2 and (r1b, c1b, r2b, c2b) != (r1, c1, r2, c2):
                is_max = False
                break
        if is_max:
            maximal.append((r1, c1, r2, c2))
    maximal.sort(key=lambda x: (x[2] - x[0] + 1) * (x[3] - x[1] + 1), reverse=True)
    result = [row[:] for row in grid]
    used = set()
    for r1, c1, r2, c2 in maximal:
        cells = set()
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                cells.add((r, c))
        if not cells & used:
            for r, c in cells:
                result[r][c] = 2
            used |= cells
    return result"""

solutions["ef135b50"] = """def solve(grid):
    from collections import deque
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    rects = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2 and not visited[r][c]:
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr2, cc2 = q.popleft()
                    comp.append((cr2, cc2))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr2 + dr, cc2 + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 2:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                min_r = min(r2 for r2, c2 in comp)
                max_r = max(r2 for r2, c2 in comp)
                min_c = min(c2 for r2, c2 in comp)
                max_c = max(c2 for r2, c2 in comp)
                rects.append((min_r, min_c, max_r, max_c))
    result = [row[:] for row in grid]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            r1a, c1a, r1b, c1b = rects[i]
            r2a, c2a, r2b, c2b = rects[j]
            row_start = max(r1a, r2a)
            row_end = min(r1b, r2b)
            if row_start <= row_end:
                if c1b < c2a:
                    left = c1b + 1
                    right = c2a - 1
                elif c2b < c1a:
                    left = c2b + 1
                    right = c1a - 1
                else:
                    continue
                if left <= right:
                    all_zero = True
                    for r in range(row_start, row_end + 1):
                        for c in range(left, right + 1):
                            if grid[r][c] != 0:
                                all_zero = False
                                break
                        if not all_zero:
                            break
                    if all_zero:
                        for r in range(row_start, row_end + 1):
                            for c in range(left, right + 1):
                                result[r][c] = 9
    return result"""

solutions["ff805c23"] = """def solve(grid):
    N = len(grid)
    ones = [(r, c) for r in range(N) for c in range(N) if grid[r][c] == 1]
    min_r = min(r for r, c in ones)
    max_r = max(r for r, c in ones)
    min_c = min(c for r, c in ones)
    max_c = max(c for r, c in ones)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    result = []
    for i in range(h):
        row = []
        for j in range(w):
            mr = N - 1 - (min_r + i)
            mc = N - 1 - (min_c + j)
            row.append(grid[mr][mc])
        result.append(row)
    return result"""

# Verify
for task_id, code in solutions.items():
    with open(f"data/arc1/{task_id}.json") as f:
        data = json.load(f)
    exec(code)
    all_pass = True
    for i, pair in enumerate(data["train"]):
        result = solve(pair["input"])
        if result != pair["output"]:
            print(f"FAIL: {task_id} train {i}")
            all_pass = False
    if all_pass:
        print(f"PASS: {task_id}")

with open("data/arc_python_solutions_retry_c.json", "w") as f:
    json.dump(solutions, f, indent=2)
print(f"Saved {len(solutions)} solutions")
