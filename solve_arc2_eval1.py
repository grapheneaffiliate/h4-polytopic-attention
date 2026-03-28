import json
import copy
from collections import Counter, defaultdict

PY = "python"
DATA_DIR = "C:/Users/atchi/h4-polytopic-attention/data/arc2"
OUT_FILE = "C:/Users/atchi/h4-polytopic-attention/data/arc2_solutions_eval1.json"

TASK_IDS = "3e6067c3,409aa875,446ef5d2,45a5af55,4a21e3da,4c3d4a41,4c416de3,4c7dc4dd,4e34c42c,53fb4810,5545f144,581f7754,58490d8a,58f5dbd5,5961cc34,5dbc8537,62593bfd,64efde09,65b59efc,67e490f4,6e453dd6,6e4f6532,6ffbe589,71e489b6,7491f3cf,7666fa5d,78332cb0,7b0280bc,7b3084d4,7b5033c1".split(",")

def load_task(tid):
    with open(f"{DATA_DIR}/{tid}.json") as f:
        return json.load(f)

def grid_eq(a, b):
    if len(a) != len(b): return False
    for r1, r2 in zip(a, b):
        if r1 != r2: return False
    return True

def test_solver(tid, solver):
    task = load_task(tid)
    for i, ex in enumerate(task['train']):
        result = solver(ex['input'])
        if not grid_eq(result, ex['output']):
            return False
    return True

def apply_solver(tid, solver):
    task = load_task(tid)
    results = []
    for tex in task['test']:
        results.append(solver(tex['input']))
    return results

# ============================================================
# SOLVER: 78332cb0
# Grid sections separated by 6-lines. Transpose between horizontal/vertical arrangement.
# ============================================================
def solve_78332cb0(grid):
    R = len(grid)
    C = len(grid[0])

    # Find separator lines (rows or cols of 6)
    sep_rows = set(r for r in range(R) if all(grid[r][c] == 6 for c in range(C)))
    sep_cols = set(c for c in range(C) if all(grid[r][c] == 6 for r in range(R)))

    # Extract section row/col ranges
    def get_ranges(total, seps):
        ranges = []
        start = None
        for i in range(total):
            if i not in seps:
                if start is None: start = i
            else:
                if start is not None:
                    ranges.append((start, i))
                    start = None
        if start is not None:
            ranges.append((start, total))
        return ranges

    row_ranges = get_ranges(R, sep_rows)
    col_ranges = get_ranges(C, sep_cols)

    # Extract sections as 2D array indexed by (row_idx, col_idx)
    sec_grid = {}
    for ri, (r1, r2) in enumerate(row_ranges):
        for ci, (c1, c2) in enumerate(col_ranges):
            sec = []
            for r in range(r1, r2):
                sec.append([grid[r][c] for c in range(c1, c2)])
            sec_grid[(ri, ci)] = sec

    n_rows = len(row_ranges)
    n_cols = len(col_ranges)

    if n_cols > 1 and n_rows == 1:
        # Horizontal sections -> stack vertically (same order)
        order = [(0, ci) for ci in range(n_cols)]
        result = []
        for i, key in enumerate(order):
            if i > 0:
                w = len(sec_grid[key][0])
                result.append([6]*w)
            result.extend(sec_grid[key])
        return result

    elif n_rows > 1 and n_cols == 1:
        # Vertical sections -> arrange horizontally (reversed order)
        order = [(ri, 0) for ri in range(n_rows-1, -1, -1)]
        sections = [sec_grid[k] for k in order]
        max_h = max(len(s) for s in sections)
        result = []
        for r in range(max_h):
            row = []
            for i, sec in enumerate(sections):
                if i > 0:
                    row.append(6)
                if r < len(sec):
                    row.extend(sec[r])
                else:
                    row.extend([7]*len(sec[0]))
            result.append(row)
        return result

    elif n_rows == 2 and n_cols == 2:
        # 2x2 grid -> stack vertically: main diagonal then anti-diagonal
        # Order: (0,0), (1,1), (0,1), (1,0)
        order = [(0,0), (1,1), (0,1), (1,0)]
        result = []
        for i, key in enumerate(order):
            if i > 0:
                w = len(sec_grid[key][0])
                result.append([6]*w)
            result.extend(sec_grid[key])
        return result

    return grid

# ============================================================
# SOLVER: 7491f3cf
# 7-row grid with border, sections separated by a column of the bg color containing 3s.
# The section with 3s in middle row gets replaced by mirror of an adjacent section.
# ============================================================
def solve_7491f3cf(grid):
    R = len(grid)
    C = len(grid[0])
    bg = grid[0][0]

    # Find separator columns (all bg)
    sep_cols = []
    for c in range(C):
        if all(grid[r][c] == bg for r in range(R)):
            sep_cols.append(c)

    # Extract sections between separators
    boundaries = [-1] + sep_cols + [C]
    sections = []
    sec_ranges = []
    for i in range(len(boundaries)-1):
        c1 = boundaries[i] + 1
        c2 = boundaries[i+1]
        if c2 > c1:
            sec = []
            for r in range(R):
                sec.append([grid[r][c] for c in range(c1, c2)])
            sections.append(sec)
            sec_ranges.append((c1, c2))

    # Find the section with the middle row (row 3) being all 3s (or a single repeated value forming separator)
    mid_row = R // 2  # row 3 for 7-row grid

    # Find which section has a "333...3" pattern in middle row
    target_idx = None
    for i, sec in enumerate(sections):
        mid = sec[mid_row]
        if len(set(mid)) == 1 and mid[0] != bg:
            # Check if this section's non-bg cells are all the same single value
            target_idx = i
            break

    if target_idx is None:
        return grid

    # The target section needs to be filled.
    # Look at the pattern: the target section has a row of separator values,
    # and the rest should be filled based on the adjacent section.
    # Actually, looking more carefully: the section to the left of the separator
    # gets its pattern reflected/copied into the section to the right.

    # Let me re-examine: each section has rows 1-5 (excluding top/bottom border).
    # The section with the uniform middle row is the "target" that needs transformation.
    # Looking at examples more carefully:

    # In train 0: section "5443445" has middle row "333333" -> becomes mirror of section to its left "5244425"
    # Wait, let me re-check by looking at the actual values.

    # Actually, the output shows the rightmost section changes.
    # The rightmost section in train 0 input: 544445 / 544445 / 544445 / 544445 / 544445
    # After: 524442 / 542424 / 544233 / 542344 / 524344 (wait that doesn't look like a simple copy)

    # Let me look again: the section with 33333 in middle is the key section.
    # Its top/bottom halves get replaced based on adjacent sections somehow.

    # Actually I think: the last section copies from the section before the 33333 one,
    # but with the 33333 section acting as a divider/mirror.

    # Let me re-examine train 0:
    # Sections: [64446|52444|25443|44544444] with separators
    # No wait - the grid is 7 rows x 25 cols. Let me think about this differently.

    # Top/bottom rows are all bg. Rows 1-5 have the pattern.
    # Middle row (row 3) has the special separator in one section.

    # In train 0 output, last section changes from 544445/544445/544445/544445/544445
    # to 524442/542424/544233/542344/524344 - wait that's wrong too.

    # Let me just look at which section changed and what it became.
    # Train 0: Input last section rows 1-5: 44445/44445/44445/44445/44445
    # Output last section rows 1-5: 24425/24245/42335/24245/24425
    # Hmm, that looks like a reflection of section index 1 (52444/54242/54244/54242/52444)
    # reversed left-right? 44425/24245/44245/24245/44425 - not quite.

    # Actually looking more carefully at the sections:
    # Section with 33333 divider: top half pattern + 33333 + bottom half pattern
    # The target (last) section gets filled with:
    # top = reverse of left-neighbor's top, bottom = reverse of left-neighbor's bottom?

    # Let me try a different approach: the last section becomes a copy of section[1]
    # (the one before the 33333 section) with left-right reversal

    # Section 1 (train 0): 24425, 42424, 42445, 42424, 24425
    # Reversed: 52442, 42424, 54424, 42424, 52442
    # Output last section: 24425, 24245, 42335, 23445, 43445 - nope

    # I think I need to recount. Let me just use a programmatic approach.

    # Actually, let me reconsider the whole thing. Looking at train 3:
    # Input:  001004200024003004000004
    #         000004020204003004000004
    #         111114002004333334000004
    #         000004020204003004000004
    #         000004200024003004000004
    # Output: 001004200024003004200024
    #         000004020204003004020204
    #         111114002004333334332334
    #         000004020204003004003004
    #         000004200024003004003004

    # So last section input: 000004/000004/000004/000004/000004 (all bg+4)
    # Wait no, let me look at this properly. Actually the sections to the right of 33333 get modified.

    # In train 3, the section after 33333 is "00400/00400/00004/00300/00300"
    # The output for that section is "20002/02020/33233/00300/00300"

    # Hmm, it looks like the section after 33333 gets filled with a version
    # of the section before 33333, but reflected through the 33333 separator,
    # and the middle row of the new section takes the values from the 33333 row inverted.

    # Let me try: the blank (all-bg) section gets replaced with the section
    # that is adjacent to the opposite side of the 33333 section.

    # In train 3:
    # Sections: [00100/00000/11111/00000/00000], [42000/02020/00200/02020/42000],
    #           [00300/00300/33333/00300/00300], [00000/00000/00004/00000/00000]
    # The last section (all bg except one 4) becomes [20002/02020/33233/00300/00300]

    # Hmm, the 33233 comes from 33333 with middle changed to 2 (= the value in section 1 at that position).

    # I think the rule is: the section after the 33333 section is a continuation/reflection.
    # The 33333 section acts as a mirror, and the section after it becomes the mirror image
    # of the section before it... but through the lens of the 33333 separator.

    # Let me look at train 0 more carefully:
    # Section before 33333: 44344/44344/33333/44344/44344
    # That IS the 33333 section itself.
    # Section 2 before that: 52444/54242/54244/54242/52444?

    # Hmm, I think I'm miscounting. Let me just code it and test by looking at actual indices.

    result = [row[:] for row in grid]

    # Find the section with uniform middle row
    target_sec = sections[target_idx]
    sep_val = target_sec[mid_row][0]

    # The section that is all bg (or mostly bg) is the one to fill
    # Find which section is "empty" (all bg or mostly bg)
    empty_idx = None
    for i, sec in enumerate(sections):
        if i == target_idx:
            continue
        non_bg = sum(1 for r in sec for v in r if v != bg)
        if non_bg == 0:
            empty_idx = i
            break

    if empty_idx is None:
        return grid

    # The source section: the one that's not target, not empty, and adjacent
    # Actually let me think about it differently.
    # There are exactly 4 sections. One has the 33333 divider, one is empty.
    # The other two have patterns. The empty one should get a transformed copy.

    # From the examples, it seems like the empty section gets filled
    # by reflecting the section on the other side of the 33333 section.

    # target_idx has 33333, empty_idx is empty.
    # If target is to the left of empty, the source is to the left of target.
    # If target is to the right of empty, the source is to the right of target.

    # Actually from train 0: sections are indexed 0,1,2,3
    # Section 2 has 33333 in middle. Section 3 is empty.
    # Source would be section 1 (to the left of target).
    # In the output, section 3 becomes: 24425/24245/42335/23445/43445? Let me check.

    # ACTUALLY: Let me look at train 3 output more carefully.
    # sections in output:
    # sec0: 00100/00000/11111/00000/00000 (unchanged)
    # sec1: 42000/02020/00200/02020/42000 (unchanged)
    # sec2: 00300/00300/33333/00300/00300 (unchanged)
    # sec3: 20002/02020/33233/00300/00300

    # So sec3 top half = reverse(sec1 top half): 42000->20002? Nope, 00024 reversed...
    # Actually 42000 reversed is 00024. But sec3 row 0 is 20002. That's not a simple reverse.

    # Hmm wait. Let me look at train 3 output sec3 more carefully:
    # 200024 / 020204 / 332334 / 003004 / 003004
    # And sec1 is: 200024 / 020204 / 002004 / 020204 / 200024

    # Oh! sec3 rows 0-1 = sec1 rows 0-1! And sec3 rows 3-4 = sec2 rows 3-4!
    # And sec3 row 2 (middle) is a blend: 332334 which is sec2[2]=333334 with sec1[2]=002004 ...
    # no that gives 332334 if you take min or overlay...

    # 333334 with 002004: where sec1 has non-bg (0,2), keep it; where sec2 has 3, keep 3; else bg.
    # 3,0->3; 3,0->3; 3,2->2; 3,0->3; 3,0->3; 4,4->4 -> 333334 not 332334

    # Hmm wait. Row 2 of sec2 is the separator (33333). In the output,
    # sec3 row 2 = 332334. That's sec1 row 2 (002004) with 0->3: 332334. Yes!
    # Where sec1 has bg (0), replace with sep_val (3). Where sec1 has non-bg, keep it.

    # And for other rows of sec3:
    # row 0: sec1 row 0 = 200024 -> sec3 row 0 = 200024 (same!)
    # row 1: sec1 row 1 = 020204 -> sec3 row 1 = 020204 (same!)
    # row 3: sec1 row 3 = 020204 -> sec3 row 3 = 003004 (different!)

    # Hmm, row 3-4 of sec3 = rows 3-4 of sec2 (003004/003004).

    # So it seems like: sec3 = top of sec1 + middle (sec1 with bg->sep_val) + bottom of sec2.
    # Where "top" = rows above mid, "bottom" = rows below mid.

    # Let me verify with train 0:
    # Need to find the actual sections. Let me just run this.

    # Actually, I think the pattern is simpler than I'm making it:
    # The empty section becomes a copy of the section that is "reflected"
    # through the separator section.

    # Specifically: for each row of the empty section:
    # - Take the corresponding row from the separator section
    # - Where the separator section has sep_val, replace with the corresponding value from another section
    # - Where the separator section has non-bg non-sep values, keep them

    # Actually maybe it's even simpler. Let me look at the big picture:
    # The last section in the output mirrors the section BEFORE the 33333 section,
    # but only in the half closer to the 33333 section.

    # I think I need to study this more carefully. Let me try a different hypothesis:
    # The empty section gets the values of the section adjacent to the 33333 section
    # on the opposite side, reflected through 33333.

    # For train 3:
    # Order: sec0, sec1, sec2(33333), sec3(empty)
    # sec3 becomes: top rows from sec1, middle from sec1 with bg->3, bottom from sec2

    # Nope. Let me try yet another approach: sec3 = overlay(sec1, sec2) where
    # sec2's non-bg values take priority.

    # sec2: 003004/003004/333334/003004/003004
    # sec1: 200024/020204/002004/020204/200024
    # overlay (sec2 priority): where sec2 != bg, use sec2; else use sec1
    # row 0: 203024 (sec2 has 003004, sec1 has 200024) -> where sec2!=0: 003004, where sec2==0: 2,0,0,2 -> 203024
    # But actual sec3 row 0 is 200024. So that's not right either.

    # OK let me try: sec3 = overlay(sec2, sec1) where sec1's non-bg values take priority.
    # Where sec1 != bg, use sec1; else use sec2.
    # row 0: sec1=200024, sec2=003004 -> 203024. Still not matching.

    # Hmm. The output sec3 = 200024/020204/332334/003004/003004.
    # This is: rows 0-1 from sec1, row 2 = sec1 with bg replaced by 3, rows 3-4 from sec2.
    # So the rule is about which HALF each row comes from:
    # - For rows above middle: take from sec1 (the one before 33333)
    # - For the middle row: take from sec1 but replace bg with sep_val
    # - For rows below middle: take from sec2 (the 33333 section itself)

    # Let me verify with train 0:
    # Order: sec0, sec1, sec2(33333), sec3(empty)
    # Wait, which section has 33333? Let me re-examine.

    # I think I really need to just look at specific indices. Let me code this differently.
    # Instead of complex logic, let me see if there's a simpler structural rule.

    # Re-examining all training examples:
    # The grid has 7 rows. Row 0 and row 6 are borders (all bg).
    # Rows 1-5 contain the patterns.
    # Row 3 (middle) of one section has all same non-bg value (the separator).

    # The pattern seems to be: the empty section gets filled by "continuing" the
    # adjacent section's pattern, with the 33333 acting as a fold/mirror.

    # Hmm, let me try: the section to the right of 33333 becomes:
    # a reflection (top-bottom flip) of the section to the LEFT of 33333,
    # but only the non-bg parts, and the middle row gets the separator value.

    # Actually wait. Let me look at this from the perspective of what the RIGHT section
    # of 33333 becomes in train 3:
    # empty->filled: row1=200024, row2=020204, row3=332334, row4=003004, row5=003004
    # sec1 (left of 33333): row1=200024, row2=020204, row3=002004, row4=020204, row5=200024
    # sec2 (the 33333 sec): row1=003004, row2=003004, row3=333334, row4=003004, row5=003004

    # filled[1] = sec1[1] = 200024 ✓
    # filled[2] = sec1[2] = 020204 ✓
    # filled[3] = sec1[3] with bg->3: 002004 -> 332334 ✓
    # filled[4] = sec2[4] = 003004 ✓
    # filled[5] = sec2[5] = 003004 ✓

    # So the rule IS: rows above mid from source (sec before 33333),
    # middle row from source with bg->sep_val, rows below mid from the 33333 section.

    # Let me verify with train 1:
    # sec2 has 33333. sec3 is to the right.
    # sec1 (before 33333): row1=883881,row2=883881,row3=333331,row4=883881,row5=883881
    # Wait, sec1 has 333331 in middle - that IS the 33333 section!

    # I think I need to reconsider which section is which.
    # Let me just use a completely different approach.

    return grid

# ============================================================
# SOLVER: 7491f3cf - REVISED
# Looking at this more carefully:
# The grid is 7 rows. Top/bottom rows are border (all bg).
# There are sections separated by bg columns.
# One section has a uniform value in its middle row (row 3) - the "separator section".
# Another section is all bg - the "empty section".
# The empty section should be filled based on the section before the separator
# and the separator section itself.
# ============================================================
def solve_7491f3cf_v2(grid):
    R = len(grid)
    C = len(grid[0])
    bg = grid[0][0]
    result = [row[:] for row in grid]
    mid = R // 2  # = 3 for 7-row grid

    # Find separator columns
    sep_cols = [c for c in range(C) if all(grid[r][c] == bg for r in range(R))]

    # Extract sections
    boundaries = [-1] + sep_cols + [C]
    sections = []
    sec_ranges = []
    for i in range(len(boundaries)-1):
        c1 = boundaries[i] + 1
        c2 = boundaries[i+1]
        if c2 > c1:
            sec = []
            for r in range(R):
                sec.append(tuple(grid[r][c] for c in range(c1, c2)))
            sections.append(sec)
            sec_ranges.append((c1, c2))

    # Find the separator section (has uniform non-bg middle row)
    sep_idx = None
    sep_val = None
    for i, sec in enumerate(sections):
        vals = set(sec[mid])
        if len(vals) == 1 and list(vals)[0] != bg:
            sep_idx = i
            sep_val = list(vals)[0]
            break

    if sep_idx is None:
        return grid

    # Find empty section (all bg)
    empty_idx = None
    for i, sec in enumerate(sections):
        if i == sep_idx: continue
        if all(v == bg for r in sec for v in r):
            empty_idx = i
            break

    if empty_idx is None:
        return grid

    # Source section: the one that's on the other side of separator from empty
    # If empty is right of separator, source is left of separator (and vice versa)
    if empty_idx > sep_idx:
        source_idx = sep_idx - 1
    else:
        source_idx = sep_idx + 1

    if source_idx < 0 or source_idx >= len(sections):
        return grid

    source = sections[source_idx]
    sep_sec = sections[sep_idx]

    # Build the new section for the empty slot:
    # For rows above mid: copy from source
    # For middle row: copy from source, replacing bg with sep_val
    # For rows below mid: copy from sep_sec

    c1, c2 = sec_ranges[empty_idx]
    w = c2 - c1

    for r in range(R):
        if r < mid:
            for j in range(w):
                result[r][c1+j] = source[r][j]
        elif r == mid:
            for j in range(w):
                v = source[r][j]
                result[r][c1+j] = sep_val if v == bg else v
        else:
            for j in range(w):
                result[r][c1+j] = sep_sec[r][j]

    return result


# ============================================================
# SOLVER: 45a5af55
# Nested concentric rectangles. Input has horizontal stripes of colors.
# Output creates nested rectangles from outside-in based on the stripe order.
# ============================================================
def solve_45a5af55(grid):
    R = len(grid)
    C = len(grid[0])

    # Each row is a single color. Find the sequence of color bands.
    bands = []
    i = 0
    while i < R:
        color = grid[i][0]
        count = 0
        while i < R and grid[i][0] == color:
            count += 1
            i += 1
        bands.append((color, count))

    # The output is a nested set of concentric rectangles.
    # The outermost band becomes the border, next band inside, etc.
    # Output size: need to figure out the dimensions.

    # From train 0: bands are (8,2),(2,5),(6,1),(8,2),(1,2),(2,2) -> output 26x26
    # From train 1: bands are (2,1),(3,1),(2,2),(3,1),(2,1),(1,1),(2,4),(8,2) -> output 24x24

    # The output has nested rectangles. Let's see:
    # Train 0 output is 26x26. Bands from outside in:
    # 8 (2 thick), 2 (5 thick), 6 (1 thick), 8 (2 thick), 1 (2 thick), 2 (2 thick)
    # Total thickness: 2+5+1+2+2+2 = 14, but output is 26 = 2*14 - 2? No, 2*13=26.
    # Each band contributes its thickness to each side, so total size = 2 * sum(thicknesses).
    # 2*(2+5+1+2+2+2) = 2*14 = 28. Not 26.
    # Wait, the innermost band fills the center, not just the border.
    # So it's 2*(sum of all but last) + last_width and last_height.
    # Hmm, but the innermost is a rectangle, so its size depends on the band thickness.

    # Actually, for concentric rectangles:
    # The total width = 2*sum(all thicknesses), total height = 2*sum(all thicknesses)
    # But the innermost fills a rectangle of width 2*last_thickness and height 2*last_thickness.
    # That's already accounted for.

    # Train 0: 2*(2+5+1+2+2+2) = 28, but output is 26x26. Off by 2.
    # Train 1: bands (2,1),(3,1),(2,2),(3,1),(2,1),(1,1),(2,4),(8,2)
    # 2*(1+1+2+1+1+1+4+2) = 2*13 = 26, but output is 24x24. Off by 2.

    # Hmm, so the formula is 2*(sum-1)? That gives 26 and 24. Yes!
    # Or equivalently: 2*sum - 2.

    # Actually wait. Let me think about this geometrically.
    # Concentric rectangles: the outermost has thickness t0 on all sides.
    # So it takes up t0 rows from top, t0 from bottom, t0 cols from left, t0 from right.
    # The remaining interior is (total_H - 2*t0) x (total_W - 2*t0).
    # For square output: side = 2*(t0+t1+...+tn).
    # But inner rectangle: the innermost band has thickness tn, so interior of second-to-last
    # is 2*tn x 2*tn, and the innermost fills that completely.
    # Total = 2*t0 + 2*t1 + ... + 2*t_{n-1} + 2*tn = 2*sum.

    # But train 0 gives 26 = 2*13, and sum = 14. So it's 2*(sum-1) = 2*13 = 26.
    # What if the innermost band thickness in the output is (tn - 0)...
    # No wait, let me count train 0 output more carefully.

    # Train 0 output 26x26:
    # 8-border: 2 thick -> outer 2 rows/cols are 8
    # 2-border: 5 thick -> next 5 rows/cols are 2
    # 6-border: 1 thick -> next 1 row/col is 6
    # 8-border: 2 thick -> next 2 rows/cols are 8
    # 1-border: 2 thick -> next 2 rows/cols are 1
    # 2-center: fills remaining 2x2 center
    # Total: 2*(2+5+1+2+2) + 2 = 2*12 + 2 = 26. Yes!
    # So innermost contributes its COUNT to each dimension (but not doubled for border).
    # Or equivalently: side = 2*(sum of all thicknesses except last) + 2*last.
    # = 2*sum. Hmm that's 2*14=28.

    # Wait no: 2*(2+5+1+2+2) = 24, then center is 2x2 -> 24+2 = 26. Yes!
    # So: side = 2*sum(all but last) + 2*last_thickness? = 24+2 = 26.
    # That's just 2*sum = 28. Still wrong.

    # 2+5+1+2+2 = 12. 2*12 = 24. Plus 2 = 26. So the center is 2 wide.
    # But last band has thickness 2, so 2*2 = 4 should be center? No.
    # Center = total - 2*sum_of_borders = 26 - 2*12 = 2. So last band fills 2x2.
    # But last band thickness is 2... so the center width = last_thickness * 2? No, it's 2 = 2*1.

    # Hmm, let me just count differently. Total side = 2*t0 + inner_side.
    # inner_side = 2*t1 + inner2_side. etc.
    # inner_last = 2*t_{n-1} + center. center is the last band.
    # So: center width/height = what?

    # Actually the last band just fills whatever space is left.
    # The total is: side = 2*(t0 + t1 + ... + t_{n-1}) + center_size.
    # And center_size is just the area filled by the last color.

    # In train 0: center_size = 2. last band is (2, 2-thick).
    # In train 1: 24 = 2*(1+1+2+1+1+1+4) + center. 2*11=22. center=2. last band is (8, 2-thick).

    # So center is always 2? That means side = 2*sum(all_thicknesses_except_last) + 2.
    # Train 0: 2*(2+5+1+2+2) + 2 = 24+2 = 26 ✓
    # Train 1: 2*(1+1+2+1+1+1+4) + 2 = 22+2 = 24 ✓

    # Actually wait. The innermost band might just have a center of 2*tn. Let me check:
    # Train 0: last band (2,2). center=2. 2*1? or 2*tn? 2*2=4 ≠ 2.

    # Maybe the center is always 2x2 (the innermost band's width from the INPUT).
    # In train 0: the last row-group has 2 rows of color 2. Input cols = 14.
    # In train 1: the last row-group has 2 rows of color 8. Input cols = 12.

    # Hmm, center is 2x2 in both cases. Last band has 2 rows in both.
    # What if center_size = min(last_rows, last_cols)? Both have 2 rows.
    # Or just center_size = last_count.
    # Train 0 last: (2, count=2). center = 2. ✓
    # Train 1 last: (8, count=2). center = 2. ✓

    # So center_size = last_band_count. And total side = 2*sum(counts[:-1]) + counts[-1]*2.
    # Wait: 2*sum(counts[:-1]) + counts[-1]?
    # Train 0: 2*(2+5+1+2+2) + 2 = 26? 2*12 + 2 = 26 ✓ (last count is 2)
    # But that's the same as 2*sum(all) - 2*last + last = 2*14 - 2*2 + 2 = 28-4+2=26. Hmm complex.
    # Or just: 2*sum(all) - last_count. 2*14-2=26 ✓. 2*13-2=24 ✓.

    # Simpler: side = 2*total_rows - last_count.
    # Train 0: R=14, last=2, 2*14-2=26 ✓
    # Train 1: R=13, last=2, 2*13-2=24 ✓
    # But wait R is actually different. Train 0 R=14, Train 1 R=13.
    # Total band thickness: 14 and 13.

    # OK I think the formula is: side = 2*R - last_count.
    # Where R is the number of input rows and last_count is the thickness of the last band.
    # Since the bands cover all R rows, sum of counts = R.
    # So side = 2*sum - last_count.

    # Let me try: side = 2*R - bands[-1][1]
    # Hmm no. Wait let me recheck: the input has C columns. The output is square.
    # If bands covered C cols similarly... but they don't, the input has uniform rows.

    # The output is square with side = 2*sum(band_counts) - band_counts[-1].
    # Train 0: 2*14 - 2 = 26 ✓
    # Train 1: 2*13 - 2 = 24 ✓

    # Wait, OR: side = 2*sum(all) - last. Hmm that doesn't feel right structurally.
    # Let me think about it as: each band contributes 2*thickness to the border,
    # EXCEPT the innermost which contributes 1*thickness (since it fills center, not just border).
    # So: side = sum(2*t_i for i in 0..n-2) + t_{n-1} = 2*sum(all) - t_{n-1}.
    # This makes structural sense! The innermost band doesn't need space on both sides.
    # It just fills the center with its thickness.

    # Actually no. The innermost band fills a rectangle. If the center is
    # t_{n-1} x t_{n-1}, then yes.

    # Hmm wait - in train 0, center_size = 2 and last band count = 2.
    # So center is 2x2, which means last band fills a 2x2 square.
    # Each band above it adds 2*thickness to the dimensions.
    # So: side = t_{n-1} + 2*sum(t_0..t_{n-2}) = t_{n-1} + 2*(sum - t_{n-1}) = 2*sum - t_{n-1}.
    # Wait that's the same formula. But doesn't quite make sense:
    # if last band fills t*t center, then inner of second-to-last = t*t,
    # and second-to-last adds 2*t_{n-2} -> 2*t_{n-2} + t.

    # Yes, it works: side = t_last + 2*(t_{n-2} + t_{n-3} + ... + t_0) = 2*sum - t_last.

    total = sum(c for _, c in bands)
    last_count = bands[-1][1]
    side = 2 * total - last_count

    # Create output grid
    out = [[0]*side for _ in range(side)]

    # Fill from outside in
    offset = 0
    for color, thickness in bands:
        # Fill the border of thickness t at current offset
        inner_side = side - 2*offset
        for t in range(thickness):
            for j in range(inner_side):
                out[offset+t][offset+j] = color
                out[offset+inner_side-1-t][offset+j] = color
                out[offset+j][offset+t] = color
                out[offset+j][offset+inner_side-1-t] = color
        offset += thickness

    return out


# ============================================================
# SOLVER: 5545f144
# Grid divided into sections by a column of a specific value.
# One section is the "clean" one (no noise), output that section.
# Actually: sections divided by separator columns. The section with
# all bg (no colored cells) or the section that differs is the output.
# ============================================================
def solve_5545f144(grid):
    R = len(grid)
    C = len(grid[0])
    bg = grid[0][0]

    # Find separator columns (single value, different from bg or same?)
    # Looking at train 0: bg=1, separator cols have values that include 3.
    # Let me find columns where all values are the same non-bg value.
    sep_cols = []
    for c in range(C):
        col = [grid[r][c] for r in range(R)]
        if len(set(col)) == 1 and col[0] != bg:
            sep_cols.append(c)

    if not sep_cols:
        # Try: separator cols where all values are same (could be bg or non-bg)
        for c in range(C):
            col = [grid[r][c] for r in range(R)]
            if len(set(col)) == 1:
                sep_cols.append(c)

    # Extract sections
    boundaries = [-1] + sep_cols + [C]
    sections = []
    for i in range(len(boundaries)-1):
        c1 = boundaries[i] + 1
        c2 = boundaries[i+1]
        if c2 > c1:
            sec = []
            for r in range(R):
                sec.append([grid[r][c] for c in range(c1, c2)])
            sections.append(sec)

    # Find the section that's all bg (or least noise) - that's the output
    # Actually looking at the task: sections have patterns, one is clean (all bg), output that.
    # Wait no - in train 0, output is 10x8 with some 4s and 1s.
    # The output matches the CLEANEST section.

    # Let me count non-bg cells per section
    counts = []
    for sec in sections:
        cnt = sum(1 for r in sec for v in r if v != bg)
        counts.append(cnt)

    # The section with fewest non-bg cells is the output
    min_idx = counts.index(min(counts))
    return sections[min_idx]


# ============================================================
# SOLVER: 6e453dd6
# Grid with a vertical separator column (col of 5 and 6).
# Pattern on left gets shifted right, and "extensions" added as 2s.
# ============================================================
def solve_6e453dd6_NEW(grid):
    R = len(grid)
    C = len(grid[0])
    bg = 6
    result = [row[:] for row in grid]

    # Find the separator column (all 5s or mix of 5 and 6)
    sep_col = None
    for c in range(C):
        col = [grid[r][c] for r in range(R)]
        if all(v == 5 for v in col):
            sep_col = c
            break
    if sep_col is None:
        for c in range(C):
            col = [grid[r][c] for r in range(R)]
            if 5 in col and all(v in [5, 6] for v in col):
                sep_col = c
                break
    if sep_col is None:
        return grid

    right_region = list(range(sep_col + 1, C))

    # Find connected components of 0s to the left of separator
    visited = set()
    components = []
    for r in range(R):
        for c in range(sep_col):
            if grid[r][c] == 0 and (r, c) not in visited:
                comp = set()
                queue = [(r, c)]
                while queue:
                    cr, cc = queue.pop()
                    if (cr, cc) in comp: continue
                    if cr < 0 or cr >= R or cc < 0 or cc >= sep_col: continue
                    if grid[cr][cc] != 0: continue
                    comp.add((cr, cc))
                    queue.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
                components.append(comp)
                visited |= comp

    for comp in components:
        max_col = max(c for r, c in comp)
        shift = (sep_col - 1) - max_col

        # Clear old positions
        for r, c in comp:
            result[r][c] = bg

        # Place at shifted positions
        for r, c in comp:
            nc = c + shift
            if 0 <= nc < sep_col:
                result[r][nc] = 0

        # Check each row for gaps (only rows that reach the separator after shift)
        rows_in_comp = sorted(set(r for r, c in comp))
        for r in rows_in_comp:
            shifted_cols = set(c + shift for rr, c in comp if rr == r)
            max_shifted = max(shifted_cols)
            if max_shifted == sep_col - 1 and len(shifted_cols) >= 2:
                min_c = min(shifted_cols)
                has_gap = any(c not in shifted_cols for c in range(min_c, max_shifted + 1))
                if has_gap:
                    for c in right_region:
                        result[r][c] = 2

    # Also handle 0-shapes on right side of separator
    right_visited = set()
    for r in range(R):
        for c in range(sep_col + 1, C):
            if grid[r][c] == 0 and (r, c) not in right_visited:
                comp = set()
                queue = [(r, c)]
                while queue:
                    cr, cc = queue.pop()
                    if (cr, cc) in comp: continue
                    if cr < 0 or cr >= R or cc <= sep_col or cc >= C: continue
                    if grid[cr][cc] != 0: continue
                    comp.add((cr, cc))
                    queue.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
                right_visited |= comp

                rows_in_comp = sorted(set(r for r, c in comp))
                for row in rows_in_comp:
                    cols = set(c for rr, c in comp if rr == row)
                    if len(cols) >= 2:
                        min_c = min(cols)
                        max_c = max(cols)
                        has_gap = any(c not in cols for c in range(min_c, max_c + 1))
                        if has_gap:
                            for cc in right_region:
                                result[row][cc] = 2

    return result

def solve_6e453dd6_OLD(grid):
    R = len(grid)
    C = len(grid[0])
    bg = 6
    result = [row[:] for row in grid]

    # Find the separator column (column of alternating 5/6 or constant 5)
    sep_col = None
    for c in range(C):
        col = [grid[r][c] for r in range(R)]
        if all(v == 5 for v in col):
            sep_col = c
            break
    if sep_col is None:
        for c in range(C):
            col = [grid[r][c] for r in range(R)]
            if all(v in [5, 6] for v in col) and 5 in col:
                sep_col = c
                break

    if sep_col is None:
        return grid

    # Looking at the pattern:
    # Each shape (group of 0s) on the left side gets shifted right by some amount
    # and the exposed right edge gets 2s.

    # Actually, looking more carefully at train 0:
    # Input left side has shapes made of 0s on bg of 6.
    # Output: the shapes move toward the separator, and where they were open
    # (touching separator side), 2s appear.

    # Let me look at it differently. The shapes are defined by 0s.
    # In the output, each shape shifts right (toward separator) so that its
    # rightmost column is adjacent to the separator.
    # And where the shape had an opening on the right, 2s fill in.

    # Actually let me look at the exact transformation:
    # Train 0 input rows (left of separator):
    # 000066666665  -> output: 666660000656
    # 006066666665  -> 666660060656
    # 600006666665  -> 666666000065
    # 000000666665  -> 666660000005
    # 666060666665  -> 666666606052
    # ...
    # It seems like: each row, the 0-pattern shifts right so the rightmost 0
    # is at position sep_col-2 (just left of separator).

    # Hmm, that's complex. Let me look at it as: the entire shape block moves right.

    # In train 0:
    # Shape 1 (rows 0-4): 0000, 0060, 0000, 0000, 6060, 6060, 6060, 6000
    # In output these move right, and where they extend past the separator boundary, 2s appear.

    # Actually I notice in the output, some cells right of separator change to 2.
    # Looking at train 0 output column to the right of separator:
    # Rows 4-6 have 2 at sep_col+1. Row 12-13 have 2 at sep_col+1.

    # The 2s appear in the column right of separator wherever a shape's row has
    # a 0 at the position adjacent to separator.

    # Let me reconsider. The transformation seems to be:
    # 1) Shift each shape so it's flush against the separator
    # 2) Where a 0 is adjacent to the separator, extend it as a 2 into the right side

    # Let me look at train 2:
    # Left side shapes are blocks of 0s. Each block has some rows with 0s.
    # In the output, the blocks shift right and 2s appear right of separator.

    # Looking at specific rows in train 2:
    # Input row 0: 6000006566 -> Output: 6600000566
    # The 0s shifted right by 1 position.
    # Input row 1: 6066606566 -> Output: 6606660522
    # The 0s shifted right by 1, and the rightmost col (before sep) got a 2 on the right of sep.

    # Input row 4: 6600006566 -> Output: 6660000566
    # Shifted right by 1.

    # So the pattern is: each connected component of 0s shifts right by some amount,
    # and where a 0 would extend past the separator, it becomes a 2 on the other side.

    # The shift amount seems to be: move right until the rightmost 0 in each block's row
    # is at sep_col - 1 (adjacent to separator).

    # But different blocks might shift different amounts.

    # Let me check: in train 2, the shapes are:
    # Block 1 (rows 0-2): occupies cols 1-5. Rightmost 0 at col 5. Sep at col 7 (the '5' col).
    # Distance to sep = 7-5-1 = 1. So shift right by 1.
    # Output row 0: 6600000566. Original: 6000006566. Yes, shifted right by 1.
    # But row 1: 6066606566 -> 6606660522. Shifted right by 1: 6606660566 but with col 7-1=6
    # being 0 adjacent to sep. The 2 appears at col 8 (right of sep). Yes!

    # Block 2 (rows 4-7): same idea.
    # Block 3 (rows 9-10): occupies col 4 only. Sep at 7. Shift right by 2.
    # Input row 9: 6666006566 -> Output: 6666600566. Shifted right by 2?
    # Original 0s at cols 4,5. Shifted to 6,7? But 7 is sep. So 0 at col 6, and col 7 stays 5.
    # Output: 6666600566. 0s at col 5,6. That's shift of 1, not 2.

    # Hmm. Let me look at it from the block perspective.
    # Block 3 rows 9-10: "66006" means 0s at positions 2,3 within the left region (cols 0-6).
    # In output: "66600" means 0s at positions 3,4. Shifted right by 1.

    # So all blocks shift right by 1 in train 2? Let me check block 4:
    # rows 12-14: 6000006 -> 6600000?
    # Input row 12: 6000006566 -> Output: 6600000566
    # Original 0s at cols 1-5. Output 0s at cols 2-6. Shifted right by 1. ✓
    # Row 13: 6060606566 -> 6606060522
    # Original 0s at cols 1,3,5. Output 0s at cols 2,4,6. Shifted right by 1.
    # And col 6 is adjacent to sep (col 7), so 2 appears at col 8. ✓

    # Block 5 rows 16-18:
    # Row 16: 6000066566 -> 6660000566
    # Original 0s at cols 1-4. Output 0s at cols 3-6. Shifted right by 2!

    # So different blocks shift different amounts!
    # Block 1,2 (width 5, cols 1-5): shift 1.
    # Block 3 (width 2, cols 4-5): shift 1.
    # Block 4 (width 5, cols 1-5): shift 1.
    # Block 5 (width 4, cols 1-4): shift 2.

    # The shift = sep_col - 1 - rightmost_0_col.
    # Block 1: rightmost 0 = 5, shift = 6-5 = 1. ✓
    # Block 4: rightmost 0 = 5, shift = 6-5 = 1. ✓
    # Block 5: rightmost 0 = 4, shift = 6-4 = 2. ✓
    # Block 3: rightmost 0 = 5, shift = 6-5 = 1. ✓

    # So the rule is: for each block of 0s, find the rightmost 0 column,
    # then shift right so that rightmost 0 is at sep_col - 1.
    # Any 0 that would go past sep_col becomes a 2 on the other side.

    # Wait, but no 0 goes past. The rightmost 0 ends up AT sep_col-1.
    # The 2s appear when a 0 is at sep_col-1 (adjacent to separator).
    # In that case, the cell at sep_col+1 in that row becomes 2.

    # Let me check: train 2 row 1: after shift, 0s at cols 2,4,6. Col 6 = sep-1.
    # So 2 appears at sep+1 = col 8. Output col 8 = 2. ✓
    # Row 0 after shift: 0s at cols 2-6. Col 6 = sep-1. So 2 at col 8?
    # But output row 0 col 8 = 5 (part of sep pattern), not 2.
    # Hmm, sep_col is the column of 5s. So sep_col+1 would be the column right of 5.

    # Wait, I need to identify the exact separator column.
    # In train 2: the separator is at col 7 (value 5 in every row, with 6 being bg).
    # Right of separator: col 8 onwards is 66 (bg).
    # In output row 0: 6600000566. Col 7=5, col 8=6, col 9=6.
    # In output row 1: 6606660522. Col 7=5, col 8=2, col 9=2.

    # So when a 0 is adjacent to separator (at col sep-1 = 6):
    # If that row has 0 at sep-1, then the cell at sep+1 becomes 2.
    # Row 0: col 6 = 0. But output col 8 = 6, not 2. So my rule is wrong!

    # Hmm. Let me re-examine. Row 0 input: 6000006566. 0s at cols 1-5.
    # Row 0 output: 6600000566. 0s at cols 2-6.
    # So col 6 is 0, adjacent to sep at col 7. But no 2 appears. Why?
    # Row 1 input: 6066606566. 0s at cols 1,3,5.
    # Row 1 output: 6606660522. 0s at cols 2,4. Cols 5,6 are 0? No: 660=6,6,0.
    # Actually output row 1: 6606660522 -> 6,6,0,6,6,6,0,5,2,2.
    # 0s at cols 2,6. Col 6 = 0. And 2s at cols 8,9.

    # Wait, 2 2s? Let me see. The right side of separator was 66 originally.
    # Now it's 22. So both positions on the right become 2.

    # In row 0, the right side stays 66 (no change). In row 1, right side becomes 22.

    # What's different? The shape in row 0 is a solid line of 0s (0,0,0,0,0).
    # The shape in row 1 has gaps (0,6,6,6,0).

    # Hmm, this doesn't quite explain it. Let me look at which rows get 2s.

    # Train 2 output right-of-separator (cols 8-9):
    # Row 0: 66 (no 2)
    # Row 1: 22 ✓
    # Row 2: 66
    # Row 3: 66
    # Row 4: 66
    # Row 5: 22 ✓
    # Row 6: 22 ✓
    # Row 7: 66
    # Row 8: 66
    # Row 9: 66
    # Row 10: 66
    # Row 11: 66
    # Row 12: 66
    # Row 13: 22 ✓
    # Row 14: 66
    # Row 15: 66
    # Row 16: 66
    # Row 17: 22 ✓
    # Row 18: 66
    # Row 19: 66

    # Now looking at which rows in the input had 0s that were NOT at the leftmost edge
    # of the shape... or maybe it's about which rows have 0 in the column adjacent to separator
    # AND also have a gap (6) somewhere in between.

    # Row 1: 0,6,6,6,0 -> has internal 6s (open shape). Gets 2. ✓
    # Row 0: 0,0,0,0,0 -> solid. No 2. ✓
    # Row 5: shape row is 0,6,0 -> has gap. Gets 2. ✓
    # Row 6: 0,6,0 -> gap. Gets 2. ✓
    # Row 7: 0,0,0 -> solid. No 2. ✓
    # Row 13: 0,6,0,6,0 -> gaps. Gets 2. ✓
    # Row 17: 0,6,6,0 -> gaps. Gets 2. ✓

    # Yes! The rule is: after shifting, if the row has any 6 (bg) between the 0s
    # (i.e., the shape is "open" / has holes in that row), then 2s appear on the right.

    # Or maybe simpler: the 2s appear where there are "openings" in the shape that
    # face the separator direction.

    # Actually, I think the 2s extend from the "mouth" of the shape through the separator.
    # The number of 2s = number of columns to the right of separator.

    # But how many 2s per row? In train 2 it's always 2 (both cols 8-9).
    # In train 0: let me check.
    # Train 0 output right of separator (sep at col ... let me find it):
    # Train 0: 16 cols. Separator is the column of 5s.

    # Wait, train 0 input row 0: 0000666666656666
    # Col 10 is 5. But that's within a shape? No: 0,0,0,0,6,6,6,6,6,6,6,5,6,6,6,6
    # The 5 is at col 11? Let me recount: 0000666666656666
    # 0(0),0(1),0(2),0(3),6(4),6(5),6(6),6(7),6(8),6(9),6(10),5(11),6(12),6(13),6(14),6(15)
    # Hmm col 11 has value 5 but other cols don't have 5 in every row.
    # Actually maybe the separator is more complex.

    # Let me look at train 0 col 11: all rows have 5 at col 11?
    # Row 0: 0000666666656666 -> col 11=5 ✓
    # Row 1: 0060666666656666 -> col 11=5 ✓
    # This seems like the separator.

    # So right of separator (cols 12-15): 4 columns.
    # Train 0 output rows 4-6:
    # Input row 4: 6660606666656666 -> Output: 6666666606052222
    # Right of sep: 2222. All 4 positions become 2.
    # Row 5: 6660606666656666 -> 6666666606052222. Same.
    # Row 6: same.
    # Row 12: 6666666600056666 -> same (no change, it's on right side).
    # Row 12 output: 6666666606052222. Hmm wait.

    # This is getting complex. Let me take a step back.

    # Actually, I think the key insight is:
    # The left side of the separator has shapes made of 0s.
    # Each shape shifts right so its bounding box's right edge touches sep-1.
    # Where the shape has "openings" (bg cells within the bounding box),
    # those openings extend through the separator as 2s to the right edge of the grid.

    # Let me verify: train 2 row 1 after shift:
    # Shape: _,0,_,_,_,0,_ (within bounding box cols 2-6, only cols 2,6 are 0, rest bg)
    # Openings at cols 3,4,5 within the shape's row span.
    # These extend through separator as 2s at cols 8-9.
    # But there are 3 openings and only 2 cols of 2s...

    # Hmm, maybe it's not about the number of openings but about the number of columns
    # right of separator.

    # Let me try yet another approach. Maybe the 2s represent the "shadow" or "projection"
    # of the interior bg cells through the separator.

    # Actually, I wonder if the rule is simply:
    # Where a row has a 0 at a position NOT adjacent to the separator,
    # AND the row also has a 0 adjacent to the separator,
    # then the cols right of separator become 2.

    # Or more simply: where a shape row has bg (6) cells between its 0s
    # (the row has both 0 and 6 in the shape area), the right side gets 2s.

    # I think this is correct for all examples I've checked. Let me code this.

    # First, find connected components of 0s.
    # For each component, find its bounding box.
    # Shift it right so rightmost 0 col = sep_col - 1.
    # For each row of the component, check if there are bg cells between 0s.
    # If so, mark the right side of separator with 2s.

    # Let me find the separator first
    left_region = list(range(sep_col))
    right_region = list(range(sep_col+1, C))

    # Find connected components of 0s in the left region
    visited = set()
    components = []

    for r in range(R):
        for c in left_region:
            if grid[r][c] == 0 and (r,c) not in visited:
                # BFS
                comp = set()
                queue = [(r,c)]
                while queue:
                    cr, cc = queue.pop()
                    if (cr,cc) in comp: continue
                    if cr < 0 or cr >= R or cc < 0 or cc >= C: continue
                    if grid[cr][cc] != 0: continue
                    if cc >= sep_col: continue
                    comp.add((cr,cc))
                    queue.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
                components.append(comp)
                visited |= comp

    # For each component, process
    for comp in components:
        rows_in_comp = sorted(set(r for r,c in comp))
        cols_in_comp = sorted(set(c for r,c in comp))
        min_col = min(cols_in_comp)
        max_col = max(cols_in_comp)

        # Shift amount
        shift = (sep_col - 1) - max_col

        # Clear old positions and write new
        for r, c in comp:
            result[r][c] = bg
        for r, c in comp:
            result[r][c + shift] = 0

        # For each row in component, check if there are bg gaps
        for r in rows_in_comp:
            row_cols = sorted(c + shift for rc, c in comp if rc == r)
            if len(row_cols) < 2:
                # Check if adjacent to separator
                if row_cols and row_cols[-1] == sep_col - 1:
                    # Single 0 at separator - no gaps, but is it open?
                    pass
                continue
            min_rc = min(row_cols)
            max_rc = max(row_cols)
            has_gap = False
            for c in range(min_rc, max_rc + 1):
                if c not in row_cols:
                    has_gap = True
                    break
            if has_gap:
                for c in right_region:
                    result[r][c] = 2

    return result


# ============================================================
# SOLVER: 7b0280bc
# Find a path/line of specific color (like 0, 2) in the grid on bg.
# Where the line is "broken" (turns back), mark the turning point
# with a different color pattern.
# Actually, this looks like: there's a spiral/path made of colors on bg 8.
# Where two parallel lines are next to each other (like 00 or 22),
# replace them with 33 or 55 (a different color).
# ============================================================
def solve_7b0280bc(grid):
    R = len(grid)
    C = len(grid[0])
    bg = grid[0][0]
    result = [row[:] for row in grid]

    # Find all non-bg values
    non_bg = {}
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg:
                non_bg[(r,c)] = grid[r][c]

    # Find the unique non-bg values
    vals = set(non_bg.values())

    # For each non-bg cell, check if it's part of a "parallel pair" that should be recolored.
    # Looking at train 0:
    # Input has a spiral path of 0s and 2s and 1s on bg 8.
    # Output: where 0s form the "outer" part, they become 5s or 3s.
    # Specifically: the beginning of the spiral (the part that forms a straight line
    # from the start before it reaches the rest of the shape) gets recolored.

    # Actually looking more carefully at train 0:
    # Input: there's a path/shape made of 0, 1, 2 values.
    # Output: some 0s become 5, some 2s become 3.
    # The 0s that become 5 are in the upper-left (the start of a spiral).
    # The 2s that become 3 are in the lower-right.

    # The spiral goes: 0->8->0 connections form the outer loop,
    # then 1->2 connections form inner parts.

    # Hmm, I think the rule is: where the path doubles back (2 cells wide instead of 1),
    # the thicker part gets recolored. The single-width parts stay.

    # Let me look at which 0s in train 0 become 5:
    # Row 3: (3,2)=0->(3,2)=5, (3,3)=0->(3,3)=5 ... these are the top-left L-shape 0s
    # Row 7: some 0s stay, some become 5.

    # Actually, looking at train 2 might be clearer.
    # In train 2, 1->3 and 6->5 transformations occur.
    # The 1s at top-left become 3s, and 6s at top-left become 5s.

    # The pattern seems to be: trace the spiral from outside. The first
    # "arm" of each color gets recolored.

    # This is getting complex. Let me look at it differently.

    # In train 0:
    # Input: 02 pairs exist (e.g., row 3: 0228, row 14-15: 0022).
    # When 0 and 2 are adjacent (in the 2-wide part of the path), they stay.
    # The parts where 0s are 2-wide (two 0s next to each other in 2x2 blocks) get recolored to 5.
    # The parts where 2s form 2x2 blocks get recolored to 3.

    # Hmm wait. In train 0 output, rows 3-4:
    # Input: 882280888088880808 / 882208888088808808
    # Output: 882280888088880858 / 882208888088808858
    # So (3,16)=0->5, (3,17)=8->5? No wait.
    # Input row 3: 882280888088880808
    # Chars: 8,8,2,2,8,0,8,8,8,0,8,8,8,8,0,8,0,8
    # Output row 3: 882280888088880858
    # Chars: 8,8,2,2,8,0,8,8,8,0,8,8,8,8,0,8,5,8
    # So (3,16): 0->5.

    # (4,16): input=8, output=8 (unchanged)
    # (4,17): input=8, output=8 (unchanged)

    # Hmm, only certain 0s become 5. Not 2x2 blocks.

    # Let me look at this more carefully with the actual connected component.
    # In train 0, the non-bg cells form some kind of spiral pattern.
    # There seem to be two spiral arms: one made of 0s and one of 2s,
    # connected by 1s at the center.

    # I think the "double-back" detection is: where two parallel lines of the same
    # non-bg value run next to each other, one of them is the "real" path and the
    # other is the "fold-back" that gets recolored.

    # This task seems quite complex. Let me skip it for now and come back.

    return grid  # placeholder


# ============================================================
# SOLVER: 7b3084d4
# Multiple small shapes on bg 0. Combine them into a single grid
# by overlaying/tiling. Output dimensions = product/sum of shape dims.
# ============================================================
def solve_7b3084d4(grid):
    R = len(grid)
    C = len(grid[0])
    bg = 0

    # Find connected components of non-bg cells
    visited = set()
    components = []

    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg and (r,c) not in visited:
                comp = set()
                queue = [(r,c)]
                while queue:
                    cr, cc = queue.pop()
                    if (cr,cc) in comp: continue
                    if cr < 0 or cr >= R or cc < 0 or cc >= C: continue
                    if grid[cr][cc] == bg: continue
                    comp.add((cr,cc))
                    queue.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
                components.append(comp)
                visited |= comp

    # Extract each component as a sub-grid
    shapes = []
    for comp in components:
        rows = [r for r,c in comp]
        cols = [c for r,c in comp]
        r1, r2 = min(rows), max(rows)
        c1, c2 = min(cols), max(cols)
        shape = []
        for r in range(r1, r2+1):
            row = []
            for c in range(c1, c2+1):
                row.append(grid[r][c])
            shape.append(row)
        shapes.append(shape)

    # Sort shapes by area (largest first? or by position?)
    # From train 0: shapes are 3x3 (9s), 3x3 (7s), 5x4 (4s), 4x3 (3s)
    # Output is 6x6.
    # 6 = 3+3 or 2*3. Hmm.

    # Actually output train 0 is:
    # 577999
    # 779999
    # 747939
    # 444333
    # 444443
    # 444333
    # So top-left 3x3 has 5/7 pattern, top-right 3x4 has 9 pattern,
    # bottom-left has 4 pattern, bottom-right has 3 pattern.

    # The shapes are arranged in a grid layout. The number of shapes determines layout.
    # 4 shapes -> 2x2 arrangement.

    # But the shapes have different sizes. Let me see:
    # Shape 1 (9s): 3 rows x 2 cols: 09,09,00,09 -> wait
    # Let me look at train 0 shapes more carefully.

    # 9s in input: (3,3)=9,(3,4)=9,(4,3)=9,(4,4)=9,(4,5)=9,(5,3)=9,(5,4)=9
    # Bounding box: rows 3-5, cols 3-5. 3x3.
    # Grid: 090/999/990

    # 7s in input: (3,14)=7,(3,15)=7,(3,16)=5-> wait, 5 is not 7.
    # Hmm: (3,13)=7,(3,14)=7,(3,15)=5 -> wait.
    # Input row 3: 00090900000007750000
    # Positions: col 3=9,4=0,5=9,6=0,... col 13=7,14=7,15=5

    # So the shapes are:
    # Shape at rows 3-5:
    #   row 3: col 3-5: 909->not contiguous?
    #   Actually 0 is bg so these aren't connected.

    # Wait, I'm confused. Let me re-read the input.
    # Row 3: 00090900000007750000
    # Non-zero: col 3=9, col 5=9, col 13=7, col 14=7, col 15=5
    # These aren't connected (separated by 0s).

    # Oh! The shapes aren't simple connected components of non-bg.
    # Some shapes have bg (0) inside them (holes).

    # Hmm but then BFS won't find them correctly. I need a different approach.

    # Let me re-examine. Maybe each shape is identified by its unique non-bg color.
    # Shape of 9s: all cells with value 9.
    # Shape of 7s: all cells with value 7 (and 5).
    # Shape of 4s: all cells with value 4.
    # Shape of 3s: all cells with value 3.

    # Wait, the 5 in row 3 is near the 7s. Maybe each shape has a unique "primary" color
    # and possibly one unique "secondary" cell.

    # Looking at output row 0: 577999. The 5 appears with 7s. So shape "7" includes the 5.
    # And shape "9" is separate.

    # OK so the shapes are:
    # In train 0:
    # Shape A (around row 3-5, cols 3-5): 9s + their bg
    # Shape B (around row 3-5, cols 13-16): 7s, 5s + bg
    # Shape C (around row 11-15, cols 3-7): 4s + bg
    # Shape D (around row 16-19, cols 17): 3s + bg

    # Extracting bounding boxes with SOME non-bg cells...
    # This seems like: find clusters of non-bg cells (maybe with small gaps).

    # Maybe I should use a different approach: find rectangular regions
    # that contain non-bg cells, separated by large gaps of bg.

    # Let me find the bounding box of each "cluster" where clusters are
    # groups of non-bg cells within a certain distance.

    # Or maybe: project onto rows and cols to find occupied bands.

    # Actually, let me just find the bounding box of each non-bg color group.
    color_cells = defaultdict(list)
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg:
                color_cells[grid[r][c]].append((r,c))

    # Group nearby color cells into clusters using proximity
    all_non_bg = [(r,c) for r in range(R) for c in range(C) if grid[r][c] != bg]

    # Use connected components with 8-connectivity and a gap tolerance
    # Or: find rectangular bounding boxes by clustering

    # Let me try: group non-bg cells into clusters where cells within distance 2 are connected
    from collections import deque
    visited = set()
    clusters = []
    for start in all_non_bg:
        if start in visited:
            continue
        cluster = set()
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            if (r,c) in cluster:
                continue
            cluster.add((r,c))
            visited.add((r,c))
            # Check neighbors within distance 2
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] != bg and (nr,nc) not in cluster:
                        queue.append((nr,nc))
        clusters.append(cluster)

    # Extract bounding boxes
    shape_grids = []
    for cluster in clusters:
        rows = [r for r,c in cluster]
        cols = [c for r,c in cluster]
        r1, r2 = min(rows), max(rows)
        c1, c2 = min(cols), max(cols)
        shape = []
        for r in range(r1, r2+1):
            row = []
            for c in range(c1, c2+1):
                row.append(grid[r][c])
            shape.append(row)
        shape_grids.append(shape)

    if len(shape_grids) != 4:
        return grid

    # Sort: top-left, top-right, bottom-left, bottom-right
    centers = []
    for cluster in clusters:
        rows = [r for r,c in cluster]
        cols = [c for r,c in cluster]
        centers.append((sum(rows)/len(rows), sum(cols)/len(cols)))

    # Sort into quadrants
    avg_r = sum(cr for cr, cc in centers) / len(centers)
    avg_c = sum(cc for cr, cc in centers) / len(centers)

    tl = tr = bl = br = None
    for i, (cr, cc) in enumerate(centers):
        if cr < avg_r and cc < avg_c: tl = i
        elif cr < avg_r and cc >= avg_c: tr = i
        elif cr >= avg_r and cc < avg_c: bl = i
        else: br = i

    if None in [tl, tr, bl, br]:
        return grid

    # Build output: stack shapes in 2x2 grid
    # Top row: tl | tr. Bottom row: bl | br.
    s_tl = shape_grids[tl]
    s_tr = shape_grids[tr]
    s_bl = shape_grids[bl]
    s_br = shape_grids[br]

    # Heights and widths
    h_top = max(len(s_tl), len(s_tr))
    h_bot = max(len(s_bl), len(s_br))
    w_left = max(len(s_tl[0]), len(s_bl[0]))
    w_right = max(len(s_tr[0]), len(s_br[0]))

    out_R = h_top + h_bot
    out_C = w_left + w_right
    out = [[bg]*out_C for _ in range(out_R)]

    # Place shapes
    # TL: aligned to top-left
    for r in range(len(s_tl)):
        for c in range(len(s_tl[0])):
            out[r][c] = s_tl[r][c]
    # TR: aligned to top-right
    for r in range(len(s_tr)):
        for c in range(len(s_tr[0])):
            out[r][w_left + c] = s_tr[r][c]
    # BL: aligned to bottom-left
    for r in range(len(s_bl)):
        for c in range(len(s_bl[0])):
            out[h_top + r][c] = s_bl[r][c]
    # BR: aligned to bottom-right
    for r in range(len(s_br)):
        for c in range(len(s_br[0])):
            out[h_top + r][w_left + c] = s_br[r][c]

    return out


# ============================================================
# SOLVER: 7b5033c1
# Trace a path through connected colored segments, output the sequence
# of colors as a 1-wide column.
# ============================================================
def solve_7b5033c1(grid):
    R = len(grid)
    C = len(grid[0])
    bg = grid[0][0]

    # Find all non-bg cells and group by connected components
    visited = set()
    segments = []

    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg and (r,c) not in visited:
                comp = set()
                color = grid[r][c]
                queue = [(r,c)]
                while queue:
                    cr, cc = queue.pop()
                    if (cr,cc) in comp: continue
                    if cr < 0 or cr >= R or cc < 0 or cc >= C: continue
                    if grid[cr][cc] != color: continue
                    comp.add((cr,cc))
                    queue.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
                segments.append((color, comp))
                visited |= comp

    # The path goes through colored segments. Each segment connects to the next.
    # The output is a column where each color appears N times (proportional to segment length).

    # Looking at train 0:
    # Segments: 1 (diagonal down-right for 4 cells), 3 (diagonal down-right for 3+cells),
    # 2 (diagonal for 3+), 4 (diagonal for 4+), 6 (3 cells)
    # Output: 1,1,1,1,1,3,3,3,3,2,2,2,2,2,4,4,4,4,4,6,6,6 (22 rows, 1 col)

    # Count per color: 1->5, 3->4, 2->5, 4->5, 6->3.
    # Segment sizes: 1 has 4 cells, 3 has 4 cells, 2 has 3 cells, 4 has 4 cells, 6 has 3 cells.
    # Hmm, that doesn't match the counts in output.

    # Let me look at the path structure. The segments form a zigzag path.
    # Each segment is a diagonal line going from one corner to another direction.

    # In train 0:
    # 1-segment: (1,5),(2,4-5),(3,5),(4,5) - L-shape?
    # Let me look at the input:
    # Row 1: 8888881888888888 -> 1 at col 6
    # Row 2: 8888811888888888 -> 1s at col 5-6
    # Row 3: 8888818888888888 -> 1 at col 5
    # Row 4: 8888818888888888 -> 1 at col 5
    # Row 5: 8888833388888888 -> 3s at cols 5-7
    # Row 6: 8888888388888888 -> 3 at col 7
    # Row 7: 8888888222888888 -> 2s at cols 7-9
    # Row 8: 8888888882888888 -> 2 at col 9
    # Row 9: 8888888882888888 -> 2 at col 9
    # Row 10: 8888884444888888 -> 4s at cols 6-9
    # Row 11: 8888884888888888 -> 4 at col 6
    # Row 12: 8888886668888888 -> 6s at cols 6-8

    # Path: starts at (1, col 6) with color 1
    # Goes: left-down (1s form an L going left then down)
    # Then right-down (3s go right then... )
    # Then right-down (2s)
    # Then left-down (4s go left then down? 4444 horizontal, then 4 below)
    # Then 6s horizontal

    # The "length" of each segment in the output seems to be the bounding box diagonal or perimeter.
    # Color 1: 5 cells in output, 5 cells in grid (positions count): (1,6),(2,5),(2,6),(3,5),(4,5) = 5 cells. ✓
    # Color 3: 4 cells in output, (5,5),(5,6),(5,7),(6,7) = 4 cells. ✓
    # Color 2: 5 cells in output, (7,7),(7,8),(7,9),(8,9),(9,9) = 5 cells. ✓
    # Color 4: 5 cells in output, (10,6),(10,7),(10,8),(10,9),(11,6) = 5 cells. ✓
    # Color 6: 3 cells in output, (12,6),(12,7),(12,8) = 3 cells. ✓

    # So each color in output appears exactly as many times as it has cells in the grid.
    # And the order follows the path from start to end.

    # Now I need to determine the order. The segments connect end-to-end.
    # Each segment's endpoint touches the next segment's start.

    # To find the order, I can build a graph of adjacencies between segments.
    # Two segments are adjacent if any cell from one is adjacent (4-connected) to a cell from the other.

    adj = defaultdict(set)
    seg_map = {}
    for i, (color, comp) in enumerate(segments):
        for r, c in comp:
            seg_map[(r,c)] = i

    for i, (color, comp) in enumerate(segments):
        for r, c in comp:
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, c+dc
                if (nr,nc) in seg_map and seg_map[(nr,nc)] != i:
                    adj[i].add(seg_map[(nr,nc)])

    # Find endpoints (segments with only 1 neighbor = ends of the chain)
    endpoints = [i for i in range(len(segments)) if len(adj[i]) <= 1]

    if not endpoints:
        return grid

    # Traverse from first endpoint
    order = []
    current = endpoints[0]
    visited_seg = set()
    while current is not None:
        order.append(current)
        visited_seg.add(current)
        next_seg = None
        for neighbor in adj[current]:
            if neighbor not in visited_seg:
                next_seg = neighbor
                break
        current = next_seg

    # Build output: column of colors, each appearing count times
    result = []
    for seg_idx in order:
        color, comp = segments[seg_idx]
        count = len(comp)
        for _ in range(count):
            result.append([color])

    return result


# ============================================================
# SOLVER: 53fb4810
# Two shapes on a bg of 8. One is a "source" pattern, the other is a "line" pattern.
# The line extends from the source through the target, continuing the source's pattern.
# ============================================================
def solve_53fb4810(grid):
    R = len(grid)
    C = len(grid[0])
    bg = 8
    result = [row[:] for row in grid]

    # Find all non-bg cells
    non_bg_cells = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg:
                non_bg_cells.add((r,c))

    # Find connected components
    visited = set()
    components = []
    for r, c in non_bg_cells:
        if (r,c) in visited:
            continue
        comp = set()
        queue = [(r,c)]
        while queue:
            cr, cc = queue.pop()
            if (cr,cc) in comp: continue
            if (cr,cc) not in non_bg_cells: continue
            comp.add((cr,cc))
            queue.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
        components.append(comp)
        visited |= comp

    # One component is a "line" (1 cell wide, repeating pattern like 23232323)
    # The other is a more complex shape.
    # The line gets extended (in both directions? or one?) and the pattern continues.

    # Actually looking at the examples:
    # Train 0: There's a horizontal line "23232323" at row 5, and a vertical line "111" shape.
    # The vertical line at row 5 connects to the horizontal one.
    # In the output, the horizontal pattern extends through the vertical shape's column.
    # And the vertical shape extends upward through the horizontal line's row? No.

    # Let me re-examine train 0:
    # Input:
    # Row 4: 88881888888888 -> 1 at col 4
    # Row 5: 88811123232323 -> 1s at 3-4, then 2,3,2,3,2,3,2,3 at cols 5-12
    # Row 6: 88881888888888 -> 1 at col 4
    # Also:
    # Row 11: 88888888884888 -> 4 at col 10
    # Row 12: 88888888882888 -> 2 at col 10
    # Row 13: 88888888881888 -> 1 at col 10
    # Row 14: 88888888811188 -> 1s at cols 9-11
    # Row 15: 88888888881888 -> 1 at col 10

    # Output:
    # Row 0: 88888888882888 -> 2 at col 10
    # Row 1: 88888888884888 -> 4 at col 10
    # ...continuing pattern 2,4,2,4 upward from the second shape
    # Row 5: 88811123234323 -> at col 10 it's now 4 (was 2)

    # So: there are two shapes. The "line shape" is the one that's a repeating 1D sequence.
    # The "complex shape" has a cross/plus structure.
    # The line extends in its direction (horizontal/vertical) to fill the grid.
    # Where it crosses the complex shape, the pattern continues.

    # More precisely: each shape has a "stem" - a line going in one direction.
    # The stem's pattern repeats to extend across the grid.

    # The second shape (rows 11-15) has a vertical stem at col 10 going: 4,2,1,1,1.
    # In the output, this stem extends upward: 2,4,2,4,2,4,2,4,2,4,4,2,1...
    # The repeating pattern is 4,2 (taken from the stem's top portion).

    # And the first shape (row 5) has a horizontal stem: 1,2,3,2,3,2,3,2,3.
    # In the output, this doesn't seem to extend (already goes to the edge).

    # I think the rule is:
    # 1. Find the "line" shape (a 1D repeating pattern).
    # 2. Extend it in its direction until it hits the "complex" shape.
    # 3. The pattern repeats.

    # Actually, let me reconsider. Looking at the output more carefully:
    # The second shape's column (col 10) gets extended upward with the pattern
    # from the repeating part of the first shape's line.

    # The first shape's line goes: 23232323. The 2 and 3 alternate.
    # The second shape's stem (col 10) goes: 4,2,1 (top to inner).
    # But in the output, above the second shape: col 10 has 2,4,2,4,2,4,2,4,2,4 going up.
    # That's 2,4 repeating - the pattern from the second shape's unique values!

    # Let me look at train 1:
    # Shape 1 (rows 0-5): cross pattern with center at ~(3,1), vertical stem going up.
    # Stem at col 1: values 8(bg),1,1,1,8,8 (rows 0-5). Core: row 1-4 has 1 at col 1.
    # Shape 2 (rows 11-16): cross pattern at ~(14,8).
    # Stem at row 5: 88... wait.

    # Actually shape 1:
    # Row 0: 8811888888888 -> 1s at cols 2-3 (that's 1,8 at 0,1 then 1,1 at 2,3)
    # Row 1: 8111188888888 -> 1s at cols 1-4
    # Row 2: 8111188888888 -> 1s at cols 1-4
    # Row 3: 8111188888888 -> 1s at cols 1-4
    # Row 4: 8811888888888 -> 1s at cols 2-3
    # Row 5: 8823888888888 -> 2 at col 2, 3 at col 3
    # Row 6-10: 8823888888888 -> same pattern repeating!

    # So the "line" from shape 1 is at cols 2-3, going down: 23,23,23,...
    # Shape 2 (rows 11-16):
    # Row 11: 8823888842888 -> 42 at cols 9-10
    # Row 12: 8823888811888 -> 11 at cols 9-10
    # Row 13: 8823888111188 -> 1111 at cols 8-11
    # Row 14: 8823888111188 -> 1111
    # Row 15: 8823888111188 -> 1111
    # Row 16: 8823888811888 -> 11

    # In output, col 9-10 above shape 2 gets 42 repeating upward:
    # Row 0: 8888888842888 -> 42 at cols 9-10
    # Row 1-10: same

    # So the stem of shape 2 (42 pattern at cols 9-10) extends upward to row 0.

    # The rule seems to be:
    # Find the two shapes. Each has a "stem" extending in one direction (the line part).
    # Extend each stem to fill the entire grid in that direction.
    # The stem is the part that juts out from the main body of the shape.

    # To implement:
    # 1. Find connected components (considering 8 as bg).
    # 2. For each component, find its "stem" - the part that extends in one direction
    #    away from the bulk of the shape.
    # 3. Extend the stem to fill the grid.

    # Let me identify stems:
    # A stem is a set of cells that form a line extending from the shape's bounding box
    # in one direction.

    # For shape 2 in train 1: main body rows 12-16, cols 8-11.
    # Stem: row 11, cols 9-10 (extends upward from the body).
    # Values: 4,2. This repeats upward.

    # For shape 1 in train 1: main body rows 0-4, cols 1-4.
    # Stem: rows 5+, cols 2-3 (extends downward).
    # Values per row: 2,3. This repeats downward.

    # So the algorithm is:
    # 1. For each component, find the main body and the stem.
    # 2. The stem is a line of cells extending from one face of the body.
    # 3. Extend the stem line to fill the grid, repeating the stem's pattern.

    # How to find the stem: it's the cells outside the "core" of the shape.
    # The core is the part that looks like a cross/plus. The stem is the line extending from it.

    # Actually, I think a simpler approach:
    # Find rows or columns that have a repeated pattern outside the main shape.
    # The "line" part is where the same values repeat at the same positions across multiple rows (or cols).

    # For each component, check if there's a consistent row pattern or column pattern
    # that appears at its edge and extends.

    # For component 1 in train 1:
    # Rows 5-10 all have: col2=2, col3=3 (and rest bg). This is the stem going down.
    # The stem pattern: [2, 3] at cols 2-3, repeating for each row.

    # For component 2 in train 1:
    # Row 11: col9=4, col10=2. Only 1 row of stem extending up.
    # The stem pattern: [4, 2] at cols 9-10.

    # To find the stem direction, look at which edge of the bbox has the "thin" extension.

    # Let me take a cleaner approach:
    # For each component, find rows and cols that have non-bg cells.
    # The "body" rows/cols are where many cells are non-bg.
    # The "stem" rows/cols are where few cells are non-bg (just the stem width).

    for comp in components:
        rows = sorted(set(r for r,c in comp))
        cols = sorted(set(c for r,c in comp))

        # Count non-bg cells per row and per col
        row_counts = Counter(r for r,c in comp)
        col_counts = Counter(c for r,c in comp)

        # Find the "body" - rows with many cells
        max_row_count = max(row_counts.values())
        body_rows = [r for r in rows if row_counts[r] >= max_row_count * 0.5]

        # Stem rows are outside the body rows
        stem_rows_above = [r for r in rows if r < min(body_rows)]
        stem_rows_below = [r for r in rows if r > max(body_rows)]

        # Similarly for columns
        max_col_count = max(col_counts.values())
        body_cols = [c for c in cols if col_counts[c] >= max_col_count * 0.5]
        stem_cols_left = [c for c in cols if c < min(body_cols)]
        stem_cols_right = [c for c in cols if c > max(body_cols)]

        # Determine stem direction
        if stem_rows_above:
            # Stem goes upward from body. Extend it further up.
            stem_row = stem_rows_above[0]  # first stem row
            stem_pattern = {}
            for r, c in comp:
                if r == stem_row:
                    stem_pattern[c] = grid[r][c]
            # Extend upward from stem to row 0
            for r in range(stem_row - 1, -1, -1):
                for c, v in stem_pattern.items():
                    result[r][c] = v
            # Also extend through the body and beyond if the pattern continues

        elif stem_rows_below:
            # Stem goes downward
            stem_row = stem_rows_below[-1]  # last stem row
            stem_pattern = {}
            for r, c in comp:
                if r == stem_row:
                    stem_pattern[c] = grid[r][c]
            # Extend downward
            for r in range(stem_row + 1, R):
                for c, v in stem_pattern.items():
                    result[r][c] = v

        elif stem_cols_left:
            # Stem goes left
            stem_col = stem_cols_left[0]
            stem_pattern = {}
            for r, c in comp:
                if c == stem_col:
                    stem_pattern[r] = grid[r][c]
            for c in range(stem_col - 1, -1, -1):
                for r, v in stem_pattern.items():
                    result[r][c] = v

        elif stem_cols_right:
            # Stem goes right
            stem_col = stem_cols_right[-1]
            stem_pattern = {}
            for r, c in comp:
                if c == stem_col:
                    stem_pattern[r] = grid[r][c]
            for c in range(stem_col + 1, C):
                for r, v in stem_pattern.items():
                    result[r][c] = v

    return result


# ============================================================
# SOLVER: 62593bfd
# Objects on bg 0 (or 5). Move objects to align based on some rule.
# Each object has an arrow indicating direction? The isolated colored cells
# might be "arrows" that indicate movement direction.
# Looking at train 0: there are shapes (made of 2,3,1) and a 4 at (4,0-1).
# Output moves shapes to edges based on the 4/arrow.
# ============================================================
def solve_62593bfd(grid):
    R = len(grid)
    C = len(grid[0])
    bg = grid[0][0]
    result = [[bg]*C for _ in range(R)]

    # Find connected components of non-bg cells
    visited = set()
    components = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg and (r,c) not in visited:
                comp = set()
                queue = [(r,c)]
                while queue:
                    cr, cc = queue.pop()
                    if (cr,cc) in comp: continue
                    if cr < 0 or cr >= R or cc < 0 or cc >= C: continue
                    if grid[cr][cc] == bg: continue
                    comp.add((cr,cc))
                    queue.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
                components.append(comp)
                visited |= comp

    # Separate "arrow" components (tiny, 1-3 cells) from "shape" components
    # Arrow has a unique color that indicates direction

    # Actually, looking at train 0 output:
    # Shape 1 (cross of 2s) moves to bottom-left.
    # Shape 2 (cross of 3s) moves to top-left.
    # Shape 3 (L of 1s) moves to top-right.
    # Arrow (4 at top-left) indicates... something.

    # In train 0:
    # Input has shapes at various positions.
    # Arrow 4 is at rows 4-5, col 0.
    # Output: all shapes move to corners/edges.

    # This is complex. Let me look at train 1:
    # bg=5. Shapes at various positions with arrows (single cells).
    # In output: shapes move to corners.

    # I think the pattern is:
    # Each shape has an associated "arrow" (single isolated cell).
    # The arrow tells the shape which corner/edge to move to.
    # Shape moves in the direction indicated by the arrow until it hits the edge.

    # To determine arrow direction from a single cell:
    # The arrow's position relative to the shape's center tells the direction.

    # Actually wait, looking at train 1 more carefully:
    # Input: 1 at (5,5) is isolated.
    # Shapes: cross of 1s at (5,3)-(9,3)...
    # Hmm this is complex.

    # Let me try a different approach. Look at train 0:
    # In the input, shapes are scattered. In the output, shapes move to edges.
    # Arrow 4 at (4,0): this shape/arrow is at the left edge. The 4 might be a color key.

    # In train 1 output, shapes move to specific corners:
    # 22255...5 at top-left -> shapes involving 2 moved to top-left
    # ...33355 at top-right -> 3s to bottom-left

    # I think the shapes don't actually move to corners. Let me compare input and output positions.

    # This task is complex. Let me skip and come back.
    return grid  # placeholder


# ============================================================
# SOLVER: 581f7754
# Each non-bg pattern has a single cell with a unique value (the "arrow").
# The pattern moves in the direction indicated by the arrow's relative position.
# ============================================================
def solve_581f7754(grid):
    R = len(grid)
    C = len(grid[0])
    bg = grid[0][0]
    result = [row[:] for row in grid]

    # Find connected components
    visited = set()
    components = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg and (r,c) not in visited:
                comp = set()
                queue = [(r,c)]
                while queue:
                    cr, cc = queue.pop()
                    if (cr,cc) in comp: continue
                    if cr < 0 or cr >= R or cc < 0 or cc >= C: continue
                    if grid[cr][cc] == bg: continue
                    comp.add((cr,cc))
                    queue.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
                components.append(comp)
                visited |= comp

    # For each component, find the "arrow" cell (unique color) and the "shape" cells
    # Then figure out direction and move shape.

    # Looking at train 0:
    # Component 1 (rows 3-4): 888/848 -> cells at (3,1-3)=8,8,8 wait that's bg.
    # Let me re-examine. bg=1 in train 0.
    # Non-bg cells: 8s and 4s.

    # Row 3: 18881111 -> 8s at cols 1-3
    # Row 4: 18481111 -> 8 at col 1, 4 at col 2, 8 at col 3... wait
    # 1,8,4,8,1,1,1,1. So non-bg: (3,1)=8,(3,2)=8,(3,3)=8,(4,1)=8,(4,2)=4,(4,3)=8? No:
    # Actually: 18481111 -> (4,0)=1(bg),(4,1)=8,(4,2)=4,(4,3)=8,(4,4)=1,...

    # Component at rows 3-4: {(3,1):8,(3,2):8,(3,3):8,(4,1):8,(4,2):4,(4,3):8}
    # This is a 2x3 shape with a 4 in the center of bottom row.
    # In output: (3,4-6)=8,8,8 and (4,4-6)=8,4,8. Moved right by 3.

    # Component at rows 8-10: (8,7)=8,(9,5-6)=8,8,4,(10,7)=8
    # Actually row 8: 11111181 -> (8,6)=8? 1,1,1,1,1,1,8,1 -> (8,6)=8.
    # Row 9: 11118841 -> (9,4)=8,(9,5)=8,(9,6)=4
    # Row 10: 11111181 -> (10,6)=8
    # Shape: a cross-ish thing with 4 at (9,6).
    # In output: row 8: (8,5)=8, row 9: (9,3-5)=8,8,4, row 10: (10,5)=8.
    # Moved left by 1.

    # The 4 is the "arrow" cell. Its position relative to shape center indicates direction.
    # In component 1: 4 at (4,2), shape center around (3.5, 2). 4 is below center -> move right?
    # In component 2: 4 at (9,6), shape center around (9, 5.5). 4 is right of center -> move... left?

    # Hmm, that doesn't work. Let me look at it differently.

    # Component 1: 888/848 = top row all 8, bottom row 8,4,8.
    # The 4 is at bottom-center. Shape moves right.
    # Component 2: shape is +sign: 8/884/8. 4 is at right. Shape moves left.

    # So the arrow (4) indicates the direction of movement? Or the opposite?
    # In comp 1: 4 at bottom -> move right?
    # In comp 2: 4 at right -> move left?

    # What about component at rows 13-16?
    # Row 13: 18181111 -> (13,1)=8,(13,3)=8
    # Row 14: 18481111 -> (14,1)=8,(14,2)=4,(14,3)=8
    # Row 15: 18181111 -> (15,1)=8,(15,3)=8
    # Row 16: 18881111 -> (16,1)=8,(16,2)=8,(16,3)=8
    # In output:
    # Row 13: 11118181 -> (13,4)=8,(13,6)=8
    # Row 14: 11118481 -> (14,4)=8,(14,5)=4,(14,6)=8
    # Row 15: 11118181 -> (15,4)=8,(15,6)=8
    # Row 16: 11118881 -> (16,4)=8,(16,5)=8,(16,6)=8
    # Moved right by 3.

    # 4 at (14,2), center around (14.5, 2). 4 is above center.
    # Shape moved right by 3.

    # Component at row 17: 11111411 -> (17,5)=4. Single cell.
    # In output: same position. Doesn't move.

    # So the single 4 doesn't move. The multi-cell shapes with 4 embedded move.

    # Looking at component 1 again: shape is 3x3-ish, 4 at bottom-center.
    # Moved right by 3 positions. The amount of movement = ?

    # In output, the shape is at cols 4-6 (was at cols 1-3). Moved right by 3.
    # Component 2: was at cols 4-6, now at cols 3-5. Moved left by 1.
    # Component 3: was at cols 1-3, now at cols 4-6. Moved right by 3.

    # These movements don't have a simple uniform rule based on the 4's position.

    # Let me look at component 1 more carefully:
    # Input (3,1-3): 888 / (4,1-3): 848
    # The 4 is at position (4,2) which is at the edge closest to...
    # The shape is 2 rows tall. 4 is in bottom row.

    # In output (3,4-6): 888 / (4,4-6): 848
    # The shape moved right until the 4 at col 5 aligns with...

    # Wait, the standalone 4 at (17,5). Could this be the "target"?
    # Component 1 has 4 at col 2, target 4 at col 5. Move right by 3. ✓
    # But component 2 has 4 at col 6, target at col 5. Move left by 1. ✓
    # Component 3 has 4 at col 2, target at col 5. Move right by 3. ✓

    # YES! The standalone 4 is the target column, and each shape moves so its
    # embedded 4 aligns with the standalone 4's column!

    # Let me verify with train 1:
    # Find the standalone non-8 single cell (or small component).
    # Train 1 has bg=8, so non-bg cells include shapes.

    # Looking for isolated single cells in train 1:
    # Row 1: 884888888333888888888 -> 4 at col 2
    # Hmm that's next to other 8s? No, bg=8. Non-bg: 4 at (1,2), 3s at (1,9-11).
    # Are these connected? col 2 to col 9 has all 8s (bg), so no.

    # Actually let me check train 2 which might be simpler.
    # Train 2: bg=3. Non-bg: 2 at (0,4). Let me check.
    # Row 0: 33332333333 -> 2 at col 4.
    # Is this isolated? (0,4)=2, neighbors: (0,3)=3(bg),(0,5)=3(bg),(1,4)=3(bg). Yes, isolated.

    # Now shapes in train 2:
    # Shape 1 (rows 3-6):
    # Row 3: 31113333333 -> 1s at cols 1-3
    # Row 4: 33133333333 -> 1 at col 2
    # Row 5: 33233333333 -> 2 at col 2
    # Row 6: 31113333333 -> 1s at cols 1-3
    # This shape has a 2 at (5,2). Target 2 is at (0,4).
    # Movement: col 2 -> col 4, so move right by 2.
    # In output:
    # Row 3: 33311133333 -> moved right by 2 ✓
    # Row 5: 33332333333 -> 2 at col 4 ✓

    # Shape 2 (rows 10-12):
    # Row 10: 33311111333 -> 1s at 3-7
    # Row 11: 33313331333 -> 1 at 3, 3s at 4-6, 1 at 7
    # Row 12: 33311121333 -> 1s at 3-5, 2 at 6, 1 at 7
    # 2 at (12,6). Target at col 4. Move left by 2.
    # Output:
    # Row 10: 31111133333 -> moved left by 2 ✓

    # Shape 3 (rows 16-18):
    # Row 16: 33331113333 -> 1s at 4-6
    # Row 17: 33311211333 -> 1 at 3, 2 at 4, 1 at 5, 1 at 6?
    # Actually: 3,3,1,1,2,1,1,3,3,3,3 -> 1 at 2, 1 at 3, 2 at 4, 1 at 5, 1 at 6
    # 2 at (17,4). Target at col 4. No movement needed.
    # Output: same position ✓

    # Great! The rule is:
    # 1. Find the standalone cell with a unique non-bg color (like 4 or 2).
    # 2. Each shape has one cell of that same color embedded in it.
    # 3. Move each shape so its embedded special cell aligns with the standalone cell.
    #    (Same column? Same row? Depends on standalone cell position.)

    # But what if the standalone is on the edge? It could indicate row OR column alignment.
    # In train 0: standalone 4 at (17,5). Shapes' 4s move to col 5. (Column alignment)
    # In train 1: standalone... let me check.

    # Train 1: bg=8.
    # What are the non-8 values? 3s and 4s and other numbers.
    # Let me find isolated single non-bg cells.

    # Hmm, I'll take a different approach. The special color is likely the one
    # that appears exactly once in most shapes AND once isolated.

    # Let me identify the "marker" color for each grid:
    # It's the non-bg color that appears in multiple components, typically as a single cell.

    # For each component, find colors that appear exactly once.
    # The marker color appears once in each shape component AND as a standalone.

    # Find isolated single cells
    single_cells = []
    for comp in components:
        if len(comp) == 1:
            r, c = list(comp)[0]
            single_cells.append((r, c, grid[r][c]))

    if not single_cells:
        return grid

    # Group markers by color
    marker_map = {}  # color -> (r, c)
    for r, c, color in single_cells:
        marker_map[color] = (r, c)

    # For each multi-cell component, find the embedded marker and compute horizontal shift
    for comp in components:
        if len(comp) == 1:
            continue

        # Find which marker color is embedded in this component
        marker_cell = None
        marker_color = None
        for r, c in comp:
            v = grid[r][c]
            if v in marker_map:
                marker_cell = (r, c)
                marker_color = v
                break

        if marker_cell is None:
            continue

        target_r, target_c = marker_map[marker_color]
        dc = target_c - marker_cell[1]  # horizontal shift only

        # Clear old positions
        for r, c in comp:
            result[r][c] = bg

        # Place at new positions
        for r, c in comp:
            nc = c + dc
            if 0 <= nc < C:
                result[r][nc] = grid[r][c]

    return result


# ============================================================
# SOLVER: 7666fa5d
# Lines radiating from points. Where lines cross, fill with 2s.
# ============================================================
def solve_7666fa5d(grid):
    R = len(grid)
    C = len(grid[0])
    bg = 8
    result = [row[:] for row in grid]

    # Find all non-bg cells (these form diagonal lines)
    non_bg = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg:
                non_bg.add((r,c))

    # Find connected components of non-bg cells using 8-connectivity (including diagonals)
    visited = set()
    components = []
    for r, c in non_bg:
        if (r,c) in visited: continue
        comp = set()
        queue = [(r,c)]
        while queue:
            cr, cc = queue.pop()
            if (cr,cc) in comp: continue
            if (cr,cc) not in non_bg: continue
            comp.add((cr,cc))
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr==0 and dc==0: continue
                    queue.append((cr+dr, cc+dc))
        components.append(comp)
        visited |= comp

    # For each pair of diagonal lines that cross, fill the crossing area with 2.
    # Actually looking at the examples:
    # Two sets of diagonal lines cross. The area between them (the "X" pattern) gets filled with 2.

    # In train 0:
    # There are two groups of diagonal lines (components).
    # Group 1: top-right area, lines going from top-left to bottom-right.
    # Group 2: bottom-left area, lines going similarly.
    # Where the lines from different groups overlap/cross, the cells become 2.

    # Actually the non-bg cells form diagonal lines (value 4 or 3).
    # In train 0: top-right diagonal (4s) and bottom-left diagonal (4s?).
    # Between them, cells become 2.

    # Let me look at train 0:
    # Input non-bg: two groups of diagonals.
    # Output: same diagonals plus 2s filling the area between them.

    # The 2s form a filled region between the diagonal lines.
    # This is like: trace each diagonal line, and between the endpoints of
    # two converging lines, fill with 2.

    # More specifically: each "fan" of diagonals originates from a point.
    # Two fans from different origins cross, and the crossing area is filled with 2.

    # In train 0:
    # Fan 1 originates from ~(0,8) going down-left and down-right.
    # Fan 2 originates from ~(15,3) going up-left and up-right.
    # The area between them (where both fans' lines would overlap) becomes 2.

    # I think the rule is:
    # Two diagonal lines meet. The area enclosed by the diagonals (between their endpoints)
    # gets filled with 2.

    # Looking at train 0 output:
    # The 2s form a diamond/triangle shape between the two groups.
    # Group 1 endpoints: (0,4), (4,8) and (0,8), (4,4) - crossing diagonals from top-right.
    # Group 2 endpoints: (8,0), (15,7) and (8,7), (15,0) - crossing from bottom-left.

    # Hmm this is complex. Let me look at it as:
    # Each diagonal line continues beyond its visible cells.
    # Where two diagonal lines from the same group converge,
    # the triangle between them fills with 2.

    # In train 0, Group 1 has lines:
    # (0,7)-(4,3) going down-left (4 cells)
    # (0,10)-(4,14) going down-right?
    # Actually let me read the values:
    # Row 0: col 7 = 4
    # Row 1: col 6 = 4, col 10 = 4, col 13 = 4
    # Hmm this isn't simple diagonals.

    # Let me look at the non-bg values:
    # Train 0 top group (rows 0-4): 4s at:
    # (0,7),(1,6),(1,10),(1,13),(2,5),(2,11),(2,14),(3,4),(3,12),(3,15),(4,7),(4,13)
    # This doesn't form simple diagonals.

    # Actually these look like MULTIPLE parallel diagonal lines.
    # Each set of parallels forms a "cone" shape.

    # In train 0:
    # Line 1: (0,7),(1,6),(2,5),(3,4) -> going down-left
    # Line 2: (1,10),(2,11),(3,12),(4,13) -> going down-right
    # Line 3: (1,13),(2,14),(3,15) -> going down-right
    # Line 4: (4,7) -> isolated?

    # Hmm this is complex. Let me look at the output to understand the pattern:
    # Output row 1: 8888888422842284 -> 4 at 7, 2s at 8-9, 8 at 10, 4 at 11, 2s at 12-13, 8, 4
    # Hmm: 4,2,2,8,4,2,2,8,4
    # So between the diagonal lines, 2s appear.

    # For train 0 specifically, the two fans are:
    # Fan A (top-right): diagonal lines going from a vertex downward.
    # Fan B (bottom-left): diagonal lines going from a vertex upward.

    # The vertex of Fan A: the point where lines converge.
    # Lines of Fan A go: (0,7)->(3,4) and (0,7)->(4,13). So vertex at top, lines go down-left and down-right.

    # I think the pattern is:
    # Each fan has an apex and two edges (diagonal lines going in opposite diagonal directions).
    # The area between the two edges (the interior of the fan) gets filled with 2.

    # For Fan A: apex at (0,7). One edge goes down-left, other goes down-right.
    # For each row below apex, the cells between the two edges become 2.

    # But there might be MULTIPLE parallel edges per fan.

    # This is very complex. Let me try a simpler approach:
    # For each pair of non-bg cells on the same diagonal, fill the cells between them with 2.

    # Actually, looking at both training examples more carefully:
    # The non-bg cells form an "X" pattern (two crossing diagonal lines).
    # Each line has parallel copies (multiple lines).
    # Between the parallel lines, 2s fill in.

    # I think the simple rule is:
    # For each row, find the non-bg cells. Between the leftmost and rightmost non-bg cell
    # in that row, fill with 2 (unless already non-bg).

    # Let me check train 0:
    # Row 1: non-bg at cols 6, 10, 13. Between 6 and 13: fill cols 7-12 with 2.
    # But output row 1: 8888888422842284.
    # Col 6=4, 7=2, 8=2, 9=8?, 10=4, 11=2, 12=2, 13=8?, 14=4
    # Wait: 8888888422842284 -> positions:
    # 0:8,1:8,2:8,3:8,4:8,5:8,6:8,7:4,8:2,9:2,10:8,11:4,12:2,13:2,14:8,15:4
    # Non-bg in input: (1,6)=4,(1,10)=4,(1,13)=4. Wait, (1,6)=4 but output col 6=8?
    # Let me re-read:
    # Input row 1: 8888888488848884 ->
    # 0:8,1:8,2:8,3:8,4:8,5:8,6:8,7:4,8:8,9:8,10:8,11:4,12:8,13:8,14:8,15:4
    # So non-bg at cols 7, 11, 15.
    # Output row 1: 8888888422842284
    # 0:8,1:8,2:8,3:8,4:8,5:8,6:8,7:4,8:2,9:2,10:8,11:4,12:2,13:2,14:8,15:4

    # So between col 7 and 11 (skip 8,9,10): cols 8,9 become 2, col 10 stays 8.
    # Between col 11 and 15: cols 12,13 become 2, col 14 stays 8.

    # This is NOT filling everything between leftmost and rightmost with 2.
    # It's filling the 2 cells adjacent to each non-bg cell, creating "bridges" between parallel lines.

    # Hmm. The 2s appear at cols 8,9 (between 7 and 11) and cols 12,13 (between 11 and 15).
    # The gap: 11-7=4, and 2s fill the middle 2 cells (8,9).
    # 15-11=4, 2s fill 12,13.

    # In row 2: input non-bg at cols 5, 11(?), and...
    # Let me read: 8888888888488848 -> col 10=4, col 14=4.
    # Wait: 8,8,8,8,8,8,8,8,8,8,4,8,8,8,4,8 -> non-bg at cols 10, 14.
    # Hmm that's only 2 non-bg cells.

    # Looking at row 0: 8888888848884888 ->
    # 0:8,1:8,2:8,3:8,4:8,5:8,6:8,7:8,8:4,9:8,10:8,11:8,12:4,13:8,14:8,15:8
    # Nope: 8888888848884888 -> col 7=4, col 11=4.
    # Actually: let me count: 8(0)8(1)8(2)8(3)8(4)8(5)8(6)8(7)4(8)8(9)8(10)8(11)4(12)8(13)8(14)8(15)
    # Hmm: 8888888848884888 -> cols: 8=4, 12=4? Yes, that matches "48884" starting at col 7.
    # Actually: char 0-6 are all 8, char 7 is 4, chars 8-10 are 8, char 11 is 8, char 12 is 4...
    # Wait: "8888888848884888" has 16 chars.
    # Positions: 0-6: 8888888, 7:4, 8-11:8888, 12:4, 13-15:888.

    # OK so the diagonal lines have spacing of 4 between them.

    # I think the two groups of lines create a grid/mesh, and the 2s fill the mesh cells.

    # This is getting very complex. Let me skip this task and move on.
    return grid  # placeholder


# ============================================================
# SOLVER: 5961cc34
# Cross pattern with 1,3 markers gets connected to an "arrow" marker.
# The cross's 3-marked edges extend in the direction of the arrow.
# ============================================================
def solve_5961cc34(grid):
    R = len(grid)
    C = len(grid[0])
    bg = 8
    result = [row[:] for row in grid]

    # Find cross shapes and the arrow marker
    # Looking at the examples:
    # There are cross/diamond shapes made of 1s and 3s on bg 8.
    # There's a 4 and 2 cell somewhere.
    # The crosses get "inflated" - the 1s become 2s and extend to the grid edge
    # following the direction of the 2/4 marker.

    # Train 0: Simple case.
    # Cross at rows 1-5, cols 1-4: 33/1111/1111/1111/11
    # Plus 4 at (9,8) and 2s at (10-12,8).
    # Output: col 8 becomes all 2s from row 0-12.

    # So the 4/2 pair indicates: 4 is the "start" and 2s extend from there.
    # In the output, all the cross patterns turn into 2s and the column of 2 extends.

    # Actually, in train 0 output, the entire column 8 is 2:
    # All rows col 8 = 2.
    # The cross shapes don't appear in the output at all!
    # Only the column of 2s remains.

    # In train 1:
    # Multiple crosses. 4 at (20,9), 2s at (21-22,9).
    # Output: the crosses get their 1-edges replaced by 2 and extended.

    # This is more subtle. In train 1 output:
    # Row 0: col 9,10 have 2s: 8888888888228888 ≠ 22888. Let me read properly.
    # 8888888888888888888228888
    # That's 25 chars. Cols 19-20 have 2,2.

    # Hmm, cols 19-20 = 22. And cols 9-10 also have values.
    # Row 3: 8888888882288888888228888
    # Cols 9-10 = 22, cols 19-20 = 22.

    # These look like the cross's "arms" extended as 2s.
    # The cross shapes have 1-thickness and 3-thickness edges.
    # In the output, the 1-edges (arms of the cross) extend as 2s to the grid edge,
    # following the direction from the 4/2 marker.

    # I think the rule is:
    # 1. Find the 4-2 pair (the arrow). This indicates a direction (from 4 toward 2).
    # 2. Each cross has edges (arms). The ones with 3s are "markers" and 1s are the "body".
    # 3. Replace 3s with the cross's "shadow" color and extend 1s as 2s in the arrow's direction.

    # Actually, I think it's simpler:
    # The 4 and 2 cells indicate a line direction.
    # In train 0: 4 at (9,8), 2 at (10,8). Direction: down (increasing row), column 8.
    # The cross body has "1" cells at specific positions.
    # In the output, replace the cross's 1s with 2s and extend the pattern downward
    # (in the direction of the arrow).

    # But in train 0, the ENTIRE grid's col 8 becomes 2. The cross is at cols 1-4.

    # Hmm wait. Let me re-read train 0 output:
    # Row 0: 888888882888888 (col 8 = 2)
    # Row 1: 888888882888888 (col 8 = 2)
    # ... all rows have col 8 = 2.

    # And the cross patterns are gone (replaced by bg).

    # So the rule for train 0 is: the cross shapes disappear, and only the 2-column remains,
    # extended to fill the entire column.

    # But what role do the cross shapes play? In train 1, the output has 2s at specific positions
    # that match the cross shapes' structures.

    # In train 1:
    # Cross 1 at rows 3-6: 33_11_1113_1111_1111_11
    # In output rows 3-6: similar but with 22 replacing some values and extending.

    # Row 3: input: 8833888881188888888338888
    # Output: 8888888882288888888228888
    # So 33 at cols 2-3 disappeared, and 22 at cols 9-10 and 19-20 appeared.

    # Row 4: input: 8111188811138888881111888
    # Output: 8888888822222222222222888

    # The 1s in the cross get replaced by 2s and the whole row between the cross arms fills with 2.

    # I think: the 3s indicate the "style" of the cross edges, and the 1s are the "substance".
    # In the output:
    # - 3s are removed (become bg)
    # - 1s become 2s
    # - The cross arms are extended in both directions to the grid edge (as 2s)
    # - The direction is toward the arrow (4/2 pair)

    # Let me check: in train 1, the arrow is at col 9-10, going down (4 at row 20, 2 at 21-22).
    # The direction is down (row increasing).

    # For cross 1 (center ~ row 5, cols ~ 5):
    # Its arms going up/down would be at col 4 extending up and down.
    # Arms going left/right would be at row 5 extending left and right.

    # But in the output, the cross arms don't quite line up with what I expect.

    # This is getting very complex. Let me focus on simpler tasks.

    return grid  # placeholder


# ============================================================
# SOLVER: 4c3d4a41
# Left side has a pattern, right side has a pattern within a 5-border box.
# The left pattern is reflected/overlaid onto the right box.
# ============================================================
def solve_4c3d4a41(grid):
    R = len(grid)
    C = len(grid[0])
    bg = 0
    result = [row[:] for row in grid]

    # The grid has two halves: left (bg=0) and right (bordered by 5s).
    # Find the 5-bordered rectangle on the right.

    # Find the bounding box of 5s
    five_rows = set()
    five_cols = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 5:
                five_rows.add(r)
                five_cols.add(c)

    if not five_rows:
        return grid

    r1 = min(five_rows)
    r2 = max(five_rows)
    c1 = min(five_cols)
    c2 = max(five_cols)

    # Interior of the 5-box: rows r1+1 to r2-1, cols c1+1 to c2-1
    # This has a pattern with non-zero non-5 values.

    # Left side: rows 0 to R-1, cols 0 to c1-1.
    # Contains a pattern of 5s forming a "mask" or "shape".

    # Looking at train 0:
    # Right box interior (rows 1-6, cols 9-18):
    # Row 1: 0302050505
    # Row 2: 0302040705
    # Row 3: 0302040705
    # Row 4: 0002040005
    # Row 5: 0000000005
    # Row 6: 0000000005
    #
    # Left side (cols 0-8):
    # Row 0: 000000000
    # Row 1: 000000000
    # Row 2: 050000050
    # Row 3: 050500050
    # Row 4: 050505050
    # Row 5: 055555550
    # Row 6: 000000000
    # Row 7: 000000000

    # The left pattern has 5s forming a triangle/staircase.
    # The right box contains the "data" and the left pattern indicates how to transform it.

    # In the output, the right box interior becomes:
    # Row 1: 0302040705
    # Row 2: 0502040505
    # Row 3: 0505040505
    # Row 4: 0505050505
    # Row 5: 0555555505
    # Row 6: 0000000005

    # Comparing input vs output right box:
    # Row 1: 0302050505 -> 0302040705 (changed cols 4-5: 05->47)
    # Row 2: 0302040705 -> 0502040505 (changed col 0: 03->05, cols 6-7: 07->05)
    #
    # This doesn't seem like a simple overlay.

    # Actually, I think the left pattern's 5s mask which rows to "shift" or "copy"
    # from the right side.

    # Hmm, let me look at this differently. The left side has a triangular staircase pattern.
    # The right side has content in its upper portion and the lower portion is empty (0s).

    # In the output, the content has "shifted down" to fill the lower portion,
    # matching the staircase pattern from the left.

    # The left pattern indicates the "filled" area of the right box.
    # Where the left has 5, the right box should have its pattern "reflected" or "shifted".

    # Actually, I think the left side shows the SHAPE of the "filled region" in the right box.
    # The right box's content gets rearranged to fill that shape.

    # In train 0 left side (the staircase):
    # Row 2: 050000050 -> 5s at cols 1, 7
    # Row 3: 050500050 -> 5s at cols 1, 3, 7
    # Row 4: 050505050 -> 5s at cols 1, 3, 5, 7
    # Row 5: 055555550 -> 5s at cols 1-7

    # This creates a triangle pointing downward-left.

    # In the output right box:
    # The non-zero non-5 values are arranged in a triangle pattern too.

    # I think: the left staircase shows the outline, and the right box's content
    # gets "reflected" so the data fills the complement of the staircase.

    # Actually, let me try a simpler interpretation:
    # The right box has 2 "layers": the non-zero content (upper rows) and zeros (lower rows).
    # The left staircase determines a line of reflection. The content gets reflected.

    # In train 0:
    # Input right box row 1: 0,3,0,2,0,5,0,5,0,5
    # Output right box row 1: 0,3,0,2,0,4,0,7,0,5
    # The 05 05 at cols 5-8 became 04 07.

    # Hmm, 04 and 07 are values that appear in row 2: 0,3,0,2,0,4,0,7,0,5.
    # So output row 1 cols 5-8 = input row 2 cols 5-8. A downward shift!

    # Output row 2: 0,5,0,2,0,4,0,5,0,5
    # Input row 2: 0,3,0,2,0,4,0,7,0,5
    # Col 0: 3->5, col 6: 7->5. These got replaced by 5.

    # Hmm, this is getting complex. Let me try a completely different approach.

    # Maybe the content inside the right box shifts/rotates.
    # Looking at the left side pattern as movement instructions:
    # Row 2 has 5s at edges (cols 1,7)
    # Row 3 adds 5 at col 3
    # Row 4 adds 5 at col 5
    # Row 5: full row of 5s

    # The staircase grows from bottom to top, left to right.
    # This looks like a "gravity" direction indicator.

    # Let me try: the content in each COLUMN of the right box shifts downward
    # until it hits the bottom border (row 5) or another value.

    # Right box columns (input):
    # Col 9 (internal col 0): 0,0,0,0,0,0 -> all 0. Output: same.
    # Col 10 (internal col 1): 3,3,3,0,0,0 -> Output: 3,5,5,5,5,0. Hmm, 5s?
    # Actually wait. Let me re-read the output more carefully.

    # I think I'm miscounting. Let me just abandon this task and try others.

    return grid  # placeholder


# ============================================================
# SOLVER: 4c7dc4dd
# Large grid with tiled pattern and two rectangular regions.
# One region has a "key" pattern, the other is the target.
# Output is extracted from the difference between them.
# ============================================================
def solve_4c7dc4dd(grid):
    R = len(grid)
    C = len(grid[0])

    # The grid has a repeating tile pattern (background).
    # Two rectangular regions differ from the tile: one has 4s bordering,
    # the other has 0s (or holes).

    # Find the tile pattern (most common small pattern that tiles the grid)
    # Then find the two rectangular regions that differ from the tile.

    # First, figure out the tile period.
    # Looking at train 0: the grid starts with 1234 repeating in cols, and rows shift.
    # It's a 4x4 tile: 1234/4123/3412/2341.

    # The two regions break this pattern.
    # Region 1 (bordered by 4s): contains the "key" pattern with non-tile values.
    # Region 2 (bordered by 1s): contains the target pattern.
    # Output = the non-tile values from region 2 (or the difference).

    # Approach:
    # 1. Find the tile period.
    # 2. Create expected tiled grid.
    # 3. Find rectangular regions where grid differs from expected.
    # 4. Extract the "anomaly" pattern from each region.
    # 5. One region has a "border" of a single value + anomaly inside.
    #    The other region has anomalies that form the output.

    # Finding tile period: try small periods.
    # Use the corners (which should be pure tile) to detect the period.

    # Find tile period using bottom-right corner (likely pure tile)
    best_tile = None
    best_match = 0
    for th in range(1, 8):
        for tw in range(1, 8):
            # Extract tile from bottom-right corner
            tile = [[grid[(R-th) + r][((C-tw) + c)] for c in range(tw)] for r in range(th)]
            # Count matches
            match = sum(1 for r in range(R) for c in range(C) if grid[r][c] == tile[r % th][c % tw])
            if match > best_match:
                best_match = match
                best_tile = (th, tw, tile)

    if best_tile is None or best_match < R * C * 0.3:
        return grid

    tile_h, tile_w, tile = best_tile

    # Create expected grid
    expected = [[tile[r % tile_h][c % tile_w] for c in range(C)] for r in range(R)]

    # Find differences
    diff_cells = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] != expected[r][c]:
                diff_cells.add((r,c))

    if not diff_cells:
        return grid

    # Find connected regions of differences (with small gap tolerance)
    visited = set()
    regions = []
    for r, c in diff_cells:
        if (r,c) in visited:
            continue
        region = set()
        queue = [(r,c)]
        while queue:
            cr, cc = queue.pop()
            if (cr,cc) in region:
                continue
            if (cr,cc) not in diff_cells:
                continue
            region.add((cr,cc))
            queue.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
        regions.append(region)
        visited |= region

    # For each region, extract the anomaly grid
    region_grids = []
    for region in regions:
        rows = sorted(set(r for r,c in region))
        cols = sorted(set(c for r,c in region))
        r1, r2 = min(rows), max(rows)
        c1, c2 = min(cols), max(cols)

        # Check if this region has a uniform border of a single value
        top_vals = set(grid[r1][c] for c in range(c1, c2+1))
        bot_vals = set(grid[r2][c] for c in range(c1, c2+1))
        left_vals = set(grid[r][c1] for r in range(r1, r2+1))
        right_vals = set(grid[r][c2] for r in range(r1, r2+1))

        border_val = None
        if len(top_vals) == 1:
            border_val = list(top_vals)[0]

        # Extract interior
        anomaly = []
        for r in range(r1+1, r2):
            row = []
            for c in range(c1+1, c2):
                v = grid[r][c]
                ev = expected[r][c]
                if v != ev:
                    row.append(v)
                else:
                    row.append(0)
            anomaly.append(row)

        region_grids.append((border_val, anomaly, (r1, c1, r2, c2)))

    # The region with 0s in its anomaly (or the "cleaner" one) is the output source.
    # Find the region with the actual non-zero anomaly pattern.

    # Actually, looking at the examples:
    # Region 1 has border of 4s and contains 0s inside.
    # Region 2 has border of another value and contains 0s/2s.
    # The output is the 0-pattern from region 2 (where 0 replaces tile values).

    # Train 0: output is 5x5 with 0s and 6/2.
    # Let me just look for the region that contains 0s (as the non-tile anomaly value).

    # Actually, I think each region has a border value and interior pattern.
    # One region acts as a "mask" (showing which cells matter) and the other
    # shows the pattern.

    # Simpler: the output is the interior of one of the regions,
    # extracting just the non-tile values.

    # From train 0 output: 00000/62222/20000/20000/20000
    # This matches the interior of region 2 after removing tile values.

    # Let me find the region whose interior, after removing tile pattern,
    # has a non-trivial pattern (not all 0s).

    for border_val, anomaly, bbox in region_grids:
        # Check if anomaly has non-zero values
        has_content = any(v != 0 for row in anomaly for v in row)
        if has_content:
            return anomaly

    # If no clear winner, return first anomaly
    if region_grids:
        return region_grids[0][1]

    return grid


# ============================================================
# SOLVER: 58490d8a
# Grid with colored shapes and a "template box" (bordered by 0s).
# Count occurrences of each shape and fill template accordingly.
# ============================================================
def solve_58490d8a(grid):
    R = len(grid)
    C = len(grid[0])
    bg = grid[0][0]

    # Find the box bordered by 0s
    zero_cells = [(r,c) for r in range(R) for c in range(C) if grid[r][c] == 0]
    if not zero_cells:
        return grid

    zr1 = min(r for r,c in zero_cells)
    zr2 = max(r for r,c in zero_cells)
    zc1 = min(c for r,c in zero_cells)
    zc2 = max(c for r,c in zero_cells)

    box_h = zr2 - zr1 + 1
    box_w = zc2 - zc1 + 1

    # Get interior colors (non-0 cells inside the box)
    colors = []
    for r in range(zr1+1, zr2):
        for c in range(zc1+1, zc2):
            if grid[r][c] != 0:
                colors.append(grid[r][c])

    # Count shapes using 8-connectivity for each color
    shape_counts = {}
    for color in set(colors):
        cells = set()
        for r in range(R):
            for c in range(C):
                if r >= zr1 and r <= zr2 and c >= zc1 and c <= zc2:
                    continue
                if grid[r][c] == color:
                    cells.add((r,c))

        visited = set()
        count = 0
        for r, c in cells:
            if (r,c) in visited:
                continue
            comp = set()
            queue = [(r,c)]
            while queue:
                cr, cc = queue.pop()
                if (cr,cc) in comp:
                    continue
                if (cr,cc) not in cells:
                    continue
                comp.add((cr,cc))
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr == 0 and dc == 0:
                            continue
                        queue.append((cr+dr, cc+dc))
            visited |= comp
            count += 1
        shape_counts[color] = count

    # Output has same dimensions as the box
    out = [[0] * box_w for _ in range(box_h)]

    # Place colors at odd rows, odd columns
    for i, color in enumerate(colors):
        row = 2 * i + 1
        count = shape_counts.get(color, 0)
        for j in range(count):
            col = 2 * j + 1
            if row < box_h and col < box_w:
                out[row][col] = color

    return out


# ============================================================
# SOLVER: 58f5dbd5
# Grid divided by borders of bg (4 or 1).
# Left columns contain "stamps" (small patterns).
# Right column is a large rectangle to be filled.
# Each stamp modifies the large rectangle.
# ============================================================
def solve_58f5dbd5(grid):
    R = len(grid)
    C = len(grid[0])
    bg = grid[0][0]

    # The grid has sections separated by bg rows/columns.
    # Looking at train 0: bg=8.
    # The right side has large colored rectangles (1s, 6s, 4s).
    # The left side has small patterns paired with each rectangle.
    # The output removes the left patterns and stamps them onto the right rectangles.

    # In train 0:
    # Right side rectangles:
    # - 1s block: rows 1-5, cols 12-17 (bordered by 8)
    # - 6s block: rows 7-11, cols 12-17
    # - 4s block: rows 13-17, cols 12-17

    # Left side patterns (rows aligned with rectangles):
    # Pattern for 1s: rows 1-3, cols 1-3: 444/484/484
    # Pattern for 6s: rows 10-11, cols 1-2: 99/89/98
    # etc.

    # In the output:
    # The 1s rectangle gets the pattern from its paired left stamp.
    # The stamp is "painted" on the rectangle, replacing 1s with the stamp colors.

    # This is complex. Let me look at the output for the 1s rectangle:
    # Output rows 1-5, cols 12-17:
    # 111118
    # 188818
    # 118118
    # 188818
    # 111118
    # 888888

    # Wait that has 8s (bg) in it. And the input stamp for 1s was:
    # 444/484/484/888 -> a 3x3 pattern of 4s and 8s.

    # The output 1s rectangle: 111118/188818/118118/188818/111118
    # This looks like the stamp pattern (4->1, 8->8) applied to the rectangle.
    # But with additional structure.

    # Hmm, the 4s in the stamp became 1s in the rectangle? And 8s stayed as 8?
    # Stamp: 444/484/484
    # Rectangle: 5 rows x 6 cols of 1s.
    # Output:
    # 111118
    # 188818
    # 118118
    # 188818
    # 111118

    # This is a symmetric pattern that looks like the stamp projected.

    # Actually, the stamp is like a "mask" or "stencil":
    # Where the stamp has bg (8), the rectangle becomes bg.
    # Where the stamp has a color (4), the rectangle keeps its own color (1).
    # But the stamp is 3x3 and the rectangle is 5x6...

    # The stamp gets centered/scaled on the rectangle? Or the border pattern gets derived.

    # Looking more carefully at the 1s rectangle output:
    # Row 0: 111118 = XXXXB
    # Row 1: 188818 = XBBXB  (B=bg)
    # Row 2: 118118 = XXBXB (X=1, B=bg)
    # Row 3: 188818 = XBBXB
    # Row 4: 111118 = XXXXB

    # The pattern: the rectangle has its border drawn with its own color (1),
    # and the interior is modified. The stamp determines the interior pattern.

    # Actually, the stamp for 1s is at rows 0-2: 444/484/484.
    # Inverting (4<->8): 888/848/848.
    # The output rectangle for 1s:
    # 11111/18881/11811/18881/11111
    # (ignoring the last col which is always 8/border)

    # This is: full border of 1, then 8,8,8 / 1,8,1 / 8,8,8 inside.
    # The inside pattern 888/181/888 matches the stamp 444/484/484 with colors swapped
    # and rotated? 444->888 (the stamp's color becomes bg), 484->181 (bg becomes the rect's color).

    # Wait: stamp 444/484/484 ->
    # Row 0: all same color (4) -> interior becomes all bg (8)
    # Row 1: 4,bg,4 -> interior becomes bg,rect_color,bg -> 8,1,8
    # Row 2: 4,bg,4 -> same

    # So the rule is: where stamp has its own color (4), the rectangle cell becomes bg (8).
    # Where stamp has bg (8), the rectangle cell keeps its color (1).
    # This is an "inverse stamp" - the stamp punches holes.

    # But the rectangle output is 5 rows, stamp is 3 rows.
    # The rectangle border stays as-is, and the stamp applies to the interior.

    # Interior of 5x6 rectangle (removing border): 3x4 inner.
    # But stamp is 3x3. Hmm.

    # Let me re-examine. The 1s rectangle is at rows 1-5, cols 12-17:
    # Actually rows 1-5 is 5 rows (1,2,3,4,5). Cols 12-17 is 6 cols.
    # Output for this block: row 1-5 col 12-17:
    # Actually output row 0 col 12-17: 111118
    # Hmm wait, in the output grid it says rows 1-5 for the 1-block.
    # Let me re-read the output:
    # Row 1: 8111118
    # Row 2: 8188818
    # Row 3: 8118118
    # Row 4: 8188818
    # Row 5: 8111118

    # So the 1s form the border of a 5x6 block, and the interior (3x4) has:
    # 8888/1881/8888

    # Stamp: 444/484/484. In binary (4=filled, 8=empty): FFF/FEF/FEF
    # Interior (rotated?): EEE/FEE/EEE? No.

    # Inner 3x4: 8888 / 1881 / 8888
    # That's: EEEE / FEEF / EEEE
    # Stamp: FFF / FEF / FEF

    # Hmm, that doesn't directly correspond. Let me re-examine what the stamp is.

    # Looking at the input left side again:
    # Row 0: 8444883838888111118
    # Wait, this is the full input row 1 or row 0?

    # I'm getting confused with row/col indexing. Let me move on to other tasks and come back.

    return grid  # placeholder


# ============================================================
# SOLVER: 71e489b6
# Grid with horizontal/vertical bands of 0s (separators) and 1s.
# Some 1-cells are 0 (holes). In the output, holes near separators
# get a 7-patch (3x3 with 7s around the 0-center).
# ============================================================
def solve_71e489b6(grid):
    R = len(grid)
    C = len(grid[0])
    result = [row[:] for row in grid]

    # Find separator lines: rows/cols that are all 0
    sep_rows = set()
    sep_cols = set()
    for r in range(R):
        if all(grid[r][c] == 0 for c in range(C)):
            sep_rows.add(r)
    for c in range(C):
        if all(grid[r][c] == 0 for r in range(R)):
            sep_cols.add(c)

    # Find "hole" cells: 0-cells that are NOT in separator rows/cols
    holes = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 0 and r not in sep_rows and c not in sep_cols:
                holes.add((r,c))

    # For each hole, determine if it's adjacent to a separator (within the 1-region).
    # If adjacent to separator, mark it with 7-pattern. If not, leave it.

    # Actually looking at train 0:
    # Holes: (0,3)=0 (in a 1-region), (2,16)=0, (4,8)=0 in the top region.
    # Also: (7,13)=0, (10,5)=0, (10,6)=0
    # Also: (14,8)=0, (15,3)=0
    # Also: (17,15)=0

    # In the output, each hole gets surrounded by 7s:
    # (0,3): output has 7 at (0,2),(0,3),(0,4) and neighboring rows.
    # Actually:
    # Output row 0: 11707111111111111 -> 7 at cols 2,4, 0 at col 3
    # Output row 1: 11777111111111177 -> 7 at cols 2,3,4 and 15,16
    # So the "patch" around (0,3) is:
    # row -1 doesn't exist, so: row 0 has 7,0,7 at cols 2-4 and row 1 has 7,7,7 at cols 2-4.
    # This is a 2x3 patch (clipped at top edge).

    # For hole (2,16):
    # row 1: cols 15,16 have 7,7
    # row 2: cols 15=1, 16=0? Actually output row 2: 11111111111111170
    # col 16 = 7, col 15 = 1. Hmm.

    # Let me re-read output row 2: 11111111111111170
    # That's 17 chars: 1*14, 1, 7, 0. Cols 15=7, 16=0.
    # Row 1: 11777111111111177 -> cols 15=7, 16=7.
    # Row 3: 11111117771111177 -> cols 15=7, 16=7.

    # So the patch around (2,16) is:
    # row 1: cols 15-16: 7,7
    # row 2: cols 15-16: 7,0
    # row 3: cols 15-16: 7,7
    # This is a 3x2 patch (clipped at right edge).

    # Wait, col 16 is the last column (grid is 17 wide).
    # The patch should be 3x3 centered on (2,16):
    # rows 1-3, cols 15-17. But col 17 doesn't exist.
    # So it's clipped: rows 1-3, cols 15-16.
    # Values: row 1: 7,7; row 2: 7,0; row 3: 7,7. Yes!

    # So the rule is: put a 3x3 patch of 7s around each hole, keeping the center as 0.
    # The patch is clipped at grid edges.

    # But wait, some holes are inside large rectangular regions of 1s
    # (not near separators), and they ALSO get 7-patches? Or only separator-adjacent ones?

    # Looking at train 0: hole at (4,8). This is not near a separator.
    # Separator rows: 5,6 (all 0s). Separator cols: none?
    # Let me check: no columns are all-0 in train 0.
    # Sep rows: let me check each row:
    # Row 5: 00000000000000000 -> yes, all 0.
    # Row 6: 00000000000000000 -> yes.
    # Row 7: 00000000000001000 -> no (has a 1 and 0s, but also 0-sep cols?).
    # Row 8: 00000000000000000 -> all 0.
    # Row 14: 00000000100000000 -> not all 0.

    # Wait: rows 5-8 are in the separator bands.
    # Row 7: 00000000000001000 -> not all 0 (has 1 at col 13). So row 7 is NOT a separator.

    # OK let me recheck. In train 0 input:
    # Row 5: 00000000000000000 -> sep
    # Row 6: 00000000000000000 -> sep
    # Row 7: 00000000000001000 -> NOT sep (has 1)
    # Row 8: 00000000000000000 -> sep
    # Row 14: 00000000100000000 -> NOT sep
    # Row 15: 00010000000000000 -> NOT sep
    # Row 16: 00000000000000000 -> sep

    # So sep rows: 5,6,8,16 (and maybe more). These are the 0-bands.

    # Holes (0-cells NOT in sep rows):
    # Row 0 col 3: hole (row 0 is NOT a sep row since it has 1s and one 0).
    # Row 2 col 16: hole.
    # Row 4 col 8: hole.
    # Row 7 col 13: this is 1, not a hole. The 0s in row 7 are in sep cols? No.
    # Actually row 7 is NOT all-0, so it's not a sep row. But it has many 0s.
    # Hmm, this cell at (7,13)=1 means it's not a hole.

    # Let me reconsider. Maybe the grid has rectangular blocks of 1s separated by 0-bands.
    # The 0-bands are rows/cols of all 0s.
    # Within each block, some 1-cells are flipped to 0 (holes).
    # These holes get 7-patches.

    # But there are also isolated 0-cells in the 0-bands (like (7,13)=1 is actually a hole
    # in the 0-band, which is a 1-cell).

    # Wait, I think I'm overcomplicating. The holes in the separator bands (like (7,13)=1)
    # are actually "holes in the 0-band" = 1-cells in a 0-region.
    # And holes in the 1-regions are 0-cells in a 1-region.

    # BOTH types of holes get 7-patches!

    # In the output:
    # (7,13): input=1 (in a 0-band). This doesn't get a 7-patch in the output.
    # (14,8): input=1 in a 0-band? Row 14: 00000000100000000. This 1 is in the 0-region.
    # Output row 14: 00000000000000000. So the 1 at (14,8) becomes 0!

    # Hmm wait. Let me re-read more carefully.

    # Train 0 output:
    # Row 14: 00000000000000000
    # Input row 14: 00000000100000000
    # So (14,8)=1 became 0. The hole (1 in 0-region) got "fixed" (removed).

    # And holes (0 in 1-region) got 7-patches.
    # And holes (1 in 0-region) got removed (set to 0).

    # So the rule is:
    # 1. The 0-bands are separator rows/cols.
    # 2. "Noise" cells: 0s in 1-regions and 1s in 0-regions.
    # 3. Noise cells in 0-regions get set to 0 (removed).
    # 4. Noise cells in 1-regions get a 3x3 patch of 7s around them.

    # But actually, looking at (7,13): input=1 in a 0-band.
    # Output (7,13): let me check. Output row 7: 00000000000000000.
    # So it became 0. ✓ (noise removed)

    # And (0,3)=0 in a 1-region -> gets 7-patch. ✓

    # First, identify which regions are "1-regions" and which are "0-regions".
    # The grid is divided into rectangular blocks by 0-bands (separator rows/cols).
    # 1-regions are blocks of 1s. 0-regions are the separators themselves.

    # But what about blocks that contain both 0s and 1s? In that case,
    # the block is a "1-region" and its 0-cells are holes.

    # I think a simpler approach:
    # Find cells that are "out of place" (different from their neighbors' majority value).
    # For isolated 0s surrounded by 1s: add 7-patch.
    # For isolated 1s surrounded by 0s: set to 0.

    # Or even simpler: the 7-patch goes around every isolated 0 in a 1-region.
    # An "isolated 0" is a 0-cell that's not part of a large 0-band.

    # Let me define "part of a 0-band" as: in a row that's all 0, or in a col that's all 0,
    # or connected to such a row/col via 0-cells.

    # Find all 0-cells that are part of the "separator network" (connected to sep rows/cols).
    sep_network = set()
    for r in sep_rows:
        for c in range(C):
            sep_network.add((r,c))
    for c in sep_cols:
        for r in range(R):
            sep_network.add((r,c))

    # BFS from sep_network through 0-cells
    queue = list(sep_network)
    while queue:
        r, c = queue.pop()
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < R and 0 <= nc < C and (nr,nc) not in sep_network and grid[nr][nc] == 0:
                sep_network.add((nr,nc))
                queue.append((nr,nc))

    # Holes in 1-regions: 0-cells NOT in sep_network
    holes_in_1 = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 0 and (r,c) not in sep_network:
                holes_in_1.add((r,c))

    # Holes in 0-regions: 1-cells in sep rows/cols (isolated 1s in 0-bands)
    holes_in_0 = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 1 and (r,c) not in sep_network:
                # Check if it's surrounded by 0s (or in a mostly-0 area)
                pass

    # Actually, I think it's simpler: just find 0-cells that are not in separator bands.
    # These are the holes that get 7-patches.

    # Also: 1-cells in separator bands need to be set to 0.

    # First, fix result by removing 1-noise in 0-bands:
    for r, c in sep_network:
        if grid[r][c] == 1:
            result[r][c] = 0

    # Now add 7-patches around holes in 1-regions:
    for r, c in holes_in_1:
        # 3x3 patch of 7s, center stays 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < R and 0 <= nc < C:
                    if (nr, nc) != (r, c):
                        if (nr, nc) not in sep_network:
                            result[nr][nc] = 7
                    # Keep center as 0

    # Also need to handle: isolated 1-cells in 0-region should be removed
    # But I think the sep_network BFS already handles finding connected 0-regions.
    # 1-cells NOT connected to any 1-region might need removal.

    # Actually, let me check: in the output, do all 1-cells in 0-bands become 0?
    # (7,13)=1 in a 0-band -> output 0. ✓
    # (14,8)=1 in a 0-band -> output 0. ✓
    # (15,3)=1 in a 0-band -> output 0. ✓

    # But these 1-cells might not be "in the sep_network" since they are 1s.
    # The sep_network only contains 0-cells + cells in sep rows/cols.
    # These 1-cells in 0-bands: are they in sep rows?
    # Row 7 is NOT a sep row (it has this 1 in it).
    # So (7,13) is NOT in sep_network and NOT a hole_in_1 (it's a 1, not a 0).

    # I need to also handle these. Let me find 1-cells that are "isolated" in 0-regions.
    # These are 1-cells not connected to any large 1-region.

    # Find connected components of 1-cells.
    visited = set()
    one_components = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 1 and (r,c) not in visited:
                comp = set()
                q = [(r,c)]
                while q:
                    cr, cc = q.pop()
                    if (cr,cc) in comp: continue
                    if cr < 0 or cr >= R or cc < 0 or cc >= C: continue
                    if grid[cr][cc] != 1: continue
                    comp.add((cr,cc))
                    q.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
                one_components.append(comp)
                visited |= comp

    # Large components are "real" 1-regions. Small components (1-2 cells) are noise.
    for comp in one_components:
        if len(comp) <= 2:
            for r, c in comp:
                result[r][c] = 0

    # Hmm but in train 0, (7,13)=1 is a single cell. It should become 0.
    # But the large 1-regions (blocks) have many cells. This threshold might work.

    # Actually wait, in the train data, the "holes in 1-regions" are also single 0-cells.
    # Those shouldn't be "removed" from the 1-region but rather get 7-patches.

    # Let me verify my approach works on train 0.

    return result


# ============================================================
# SOLVER: 409aa875
# Grid of 7s with some 9s/0s forming a pattern.
# The pattern has "antennae" (lines of the pattern value extending from a corner).
# Some antennae are missing. Fill them in based on the symmetric position.
# ============================================================
def solve_409aa875(grid):
    R = len(grid)
    C = len(grid[0])
    bg = 7
    result = [row[:] for row in grid]

    # Find non-bg cells
    non_bg = {}
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg:
                non_bg[(r,c)] = grid[r][c]

    # Looking at the examples:
    # Train 0: grid has 9s forming a pattern (L-shape at bottom rows 14-15).
    # There's a 9 at (3,12), (4,12-13). These are "antennae".
    # In the output, new cells appear at (9,1) with value 1.
    # So 1 = replacement value for the missing antenna mirror.

    # Actually, let me look at this differently.
    # The bottom two rows (14-15) have a pattern of 9s and 0s.
    # Row 14: 7977777977777797
    # Row 15: 9797779797777979

    # The top portion has some 9s at (3,12),(4,12-13). These relate to the pattern.

    # In the output, (9,7) gets value 1. The 1 seems special.

    # I think the bottom rows encode a "code" for the columns.
    # Each column pair in the bottom rows has a 2-bit value:
    # Col 0: rows 14,15 = 7,9 -> pattern 01
    # Col 1: rows 14,15 = 9,7 -> pattern 10
    # etc.

    # The 9s in the main area form shapes. Each shape corresponds to one column's "code".
    # If the code matches, a color/value appears.

    # This is very abstract. Let me look at train 1 and 2 to find commonalities.

    # Train 1 has 0s instead of 9s and similar bottom rows with 0s and 7s.
    # Train 2 has 2s instead and similar structure.

    # I think the bottom rows have a pattern of the shape's color (9/0/2) and bg (7).
    # Each column in the bottom has a "signal" (present or absent).
    # The main grid area has the shape in certain positions.
    # The output adds new cells based on some reflection/symmetry.

    # Looking at train 2 output:
    # Row 0: 7777777777777777 -> no change
    # Row 1: 9779779779777777 -> 9s appeared at cols 0, 3, 6, 9
    # etc.

    # The new 9s in the output seem to be reflections/extensions of the existing pattern.

    # Actually, I notice in train 2:
    # Bottom rows (14-15):
    # 7777722722722722
    # 7777727727727727
    # So the pattern is in cols 5-15: 22722722722
    # /27727727727

    # The main area has two groups:
    # Group 1: rows 6-7, cols 5-15: same as bottom rows!
    # Group 2: rows 11-12, cols 13-15: 227/277

    # In output: rows 1 and 9 get filled: 9779779779777777 and 9779779779777777.
    # These are at distance 5 from rows 6 and 14 respectively.

    # Hmm, this is complex. Let me skip this task.

    return grid  # placeholder


# ============================================================
# SOLVER: 6e4f6532
# Grid with horizontal/vertical bands of border colors (11,22,33,44,77).
# Interior cells have patterns that need to be reorganized/reflected.
# ============================================================
def solve_6e4f6532(grid):
    R = len(grid)
    C = len(grid[0])
    result = [row[:] for row in grid]

    # The grid has bands/stripes at the top, bottom, left, right.
    # These are constant (all same value in each band).
    # The interior has two "windows" separated by a band.
    # Each window has a pattern that needs to be processed.

    # Looking at train 0:
    # Left band: cols 0-1, all 1s.
    # Right band: cols 24-25, all 2s.
    # Middle band: cols 12-13, all 4s.
    # Left window: cols 2-11, rows 0-12.
    # Right window: cols 14-23, rows 0-12.

    # Left window has some shapes (8s, 5s, 2s, etc.).
    # Right window has different shapes.

    # In the output:
    # The shapes within each window are "moved" - specifically, they sink to the bottom
    # (or move toward the center band).

    # Looking at left window:
    # Input: shapes at rows 4-10 (various positions).
    # Output: shapes shifted to rows 2-9 approximately.

    # Actually, the shapes seem to be reflected or gravity-applied.

    # In left window input:
    # Row 3: 555545(5555) -> 4 at col 7, 5 at col 8
    # Actually: 5555455554 (cols 2-11): 5,5,5,5,4,5,5,5,5,4
    # Wait, those are 5s with a 4 at position 4.

    # Hmm, the 5 is the bg of the left window. And shapes within are made of 8, 2, 9, etc.

    # Looking at specific non-5 cells in the left window:
    # (4,4-8): 88855 -> 8,8,8,5,5
    # (5,3-8): 889855 -> 8,8,9,8,5,5?

    # This is getting complex. Let me look at the shapes' movement.

    # I think the rule involves "gravity" - shapes fall toward a specific edge.
    # In the left window, shapes move toward the middle band (right).
    # In the right window, shapes move toward the middle band (left).

    # Or: the shapes within each window get reflected across the window's center axis.

    # Looking at train 0 left window:
    # Input has shapes in upper-left area. Output has shapes in upper-right area (closer to center band).

    # Let me check train 1:
    # It has horizontal bands instead of vertical.
    # Top band: rows 0-1, all 2s. Bottom band: rows 24-25, all 3s.
    # Middle band: ? Let me check.

    # Looking at the structure, this is quite complex. Let me skip.

    return grid  # placeholder


# ============================================================
# SOLVER: 6ffbe589
# Grid with shapes. The shape has a "frame" and interior pattern.
# There are "direction markers" (isolated colored cells).
# The shape gets moved/reflected to align with the markers.
# Actually: the output is just the shape itself, extracted and possibly cleaned up.
# ============================================================
def solve_6ffbe589(grid):
    R = len(grid)
    C = len(grid[0])
    bg = 0

    # Find the frame (rectangle of 3s or border values)
    # Looking at train 0: there's a rectangle bordered by 3s.
    # The frame has a notch/extension on one side.
    # Output = the frame with its interior, centered.

    # Find all 3-cells
    three_cells = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 3:
                three_cells.add((r,c))

    if not three_cells:
        # Try finding the frame color differently
        # Look for rectangular border
        pass

    # Find bounding box of 3-cells
    r1 = min(r for r,c in three_cells)
    r2 = max(r for r,c in three_cells)
    c1 = min(c for r,c in three_cells)
    c2 = max(c for r,c in three_cells)

    # Extract the region
    out = []
    for r in range(r1, r2+1):
        row = []
        for c in range(c1, c2+1):
            row.append(grid[r][c])
        out.append(row)

    return out


# ============================================================
# SOLVER: 446ef5d2
# Multiple small rectangles with different patterns.
# They combine into a single larger rectangle.
# ============================================================
def solve_446ef5d2(grid):
    R = len(grid)
    C = len(grid[0])
    bg = 8

    # Find all non-bg rectangular regions
    # These are bordered by the first non-bg value found.
    # Each region is a rectangle of 7s containing a pattern.

    # Looking at train 0:
    # There are multiple 7-bordered rectangles, each containing a pattern.
    # There are also some "loose" colored cells (like 4,8 at bottom-right).

    # The output combines all rectangles into one big rectangle,
    # with the patterns merged.

    # I think the rectangles overlap/tile. The non-bg content from each
    # is placed in the combined rectangle.

    # Let me find the 7-bordered rectangles.

    # Find connected components of 7s
    visited = set()
    seven_comps = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 7 and (r,c) not in visited:
                comp = set()
                queue = [(r,c)]
                while queue:
                    cr, cc = queue.pop()
                    if (cr,cc) in comp: continue
                    if cr < 0 or cr >= R or cc < 0 or cc >= C: continue
                    if grid[cr][cc] != 7: continue
                    comp.add((cr,cc))
                    queue.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
                seven_comps.append(comp)
                visited |= comp

    # Each 7-component's bounding box is a rectangle.
    # The interior of each rectangle has a pattern.

    # Actually, looking at train 0 more carefully:
    # The rectangles share some edges (they're adjacent).
    # Rectangle 1: rows 0-3, cols 8-13 (7-border, interior has 0s and 7s)
    # Rectangle 2: rows 4-7, cols 0-3 (7-border, interior has 0s)
    # etc.

    # In the output, all rectangles merge into one:
    # rows 3-9, cols 5-17 (a single 7-bordered rectangle with all patterns inside).

    # The "loose" colored cells (4 and 8 at bottom-right corner: rows 9-10, cols 16-18)
    # act as "connectors" indicating how rectangles join.

    # This is complex. I'll need to understand the joining rule better.
    # Let me check train 1:
    # Input has 4 rectangles. Output has 1.
    # The loose cells are at specific positions indicating the arrangement.

    # Actually, I think the loose colored cells (non-7, non-0, non-8-bg)
    # indicate the position/direction for assembly.

    # In train 0: loose cells 4,8 at (9,16)=4,(9,17)=4,(10,16)=4,(10,17)=4.
    # Wait, that's a 2x2 block of 4s. And there's 8 nearby.

    # I think the rectangles represent "views" of a 3D object from different sides.
    # Or they're "unfolded" faces of a box.

    # Actually, let me look at this as: the rectangles have shared edges/borders
    # (made of 7s). When two rectangles share an edge, they fold together.
    # The loose colored cells mark the "fold" directions.

    # This is very complex. Let me skip for now.

    return grid  # placeholder


# ============================================================
# SOLVER: 4e34c42c
# Grid with small shapes. Each shape has a "border" color and "interior" pattern.
# The shapes combine by having their borders overlap.
# Output = combined shape.
# ============================================================
def solve_4e34c42c(grid):
    R = len(grid)
    C = len(grid[0])
    bg = grid[0][0]

    # Find all non-bg connected components (each shape)
    visited = set()
    components = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg and (r,c) not in visited:
                comp = set()
                queue = [(r,c)]
                while queue:
                    cr, cc = queue.pop()
                    if (cr,cc) in comp: continue
                    if cr < 0 or cr >= R or cc < 0 or cc >= C: continue
                    if grid[cr][cc] == bg: continue
                    comp.add((cr,cc))
                    queue.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
                components.append(comp)
                visited |= comp

    # Extract each component as a subgrid
    shapes = []
    for comp in components:
        rows = sorted(set(r for r,c in comp))
        cols = sorted(set(c for r,c in comp))
        r1, r2 = min(rows), max(rows)
        c1, c2 = min(cols), max(cols)
        shape = []
        for r in range(r1, r2+1):
            row = []
            for c in range(c1, c2+1):
                row.append(grid[r][c])
            shape.append(row)
        shapes.append((shape, r1, c1))

    # Looking at train 0:
    # 3 shapes. Each is a "framed" rectangle.
    # Shape 1 (row 6-10): 5x5 with 9-border and interior.
    # Shape 2 (rows 11-15): 5x5 with 6-border and interior.
    # Shape 3 (rows 18-20): 3x3 with 3-border and interior.

    # Output: 5x12. This is shapes 1 and 2 and 3 arranged side by side?
    # Or overlapping?

    # Output:
    # 689888888888
    # 611188282333
    # 614111222323
    # 611188282333
    # 689888888888

    # This looks like shapes placed side by side horizontally:
    # Left part (5x3): 689/611/614/611/689
    # Middle part (5x6): 888888/188282/111222/188282/888888
    # Right part (5x3): 888/333/323/333/888

    # These correspond to the 3 shapes:
    # Shape 3 (3x3): 689/611/614/611/689 -> but shape 3 is 3x3: 333/323/333 (3-rows)
    # Actually shape 3 from input:
    # Row 18: 333, Row 19: 323, Row 20: 333

    # Hmm, the left part of output (689/611/614/611/689) is 5x3:
    # This looks like shape 2 (the 6-border):
    # Row 11: 68988 -> 6,8,9,8,8 -> just the left 3 cols: 6,8,9
    # But 689 matches! So shape 2 is placed on the left.

    # And middle part: 888888/188282/111222/188282/888888
    # Shape 1 was:
    # 98888/81188/84111/81188/98888 (5x5 with 9-border)
    # The middle would be: 88888/11882/41112/11882/88888 (inner 5x5 without left col?)

    # Actually I think the shapes are being "interlocked":
    # Each shape has a border and an interior extending in a direction.
    # The "9" part of shape 1 and the "6" part of shape 2 overlap.

    # The output is shapes 2, 1, and 3 concatenated with their borders overlapping.

    # Shape 2 (5x5): 689/611/614/611/689 (first 3 cols of output match exactly!)
    # Wait let me re-read shape 2:
    # Input rows 11-15:
    # 88888888888888888868988
    # 88888888888888888861188
    # 88888888888888888861488
    # 88888888888888888861188
    # 88888888888888888868988

    # Extracting non-bg: cols 18-22:
    # 68988, 61188, 61488, 61188, 68988

    # Output left 5 cols:
    # 68988, 61118, 61411, 61118, 68988 -> hmm not matching

    # Actually output:
    # 689888888888
    # 611188282333
    # 614111222323
    # 611188282333
    # 689888888888

    # Left 3 cols: 689/611/614/611/689. This is shape 2's left 3 cols (68,61,61,61,68 -> no).
    # Shape 2 left col: 6,6,6,6,6. Second col: 8,1,1,1,8. Third: 9,1,4,1,9.
    # Left 3 cols of output: 6,8,9 / 6,1,1 / 6,1,4 / 6,1,1 / 6,8,9.
    # That matches shape 2! ✓

    # Now shape 1 (rows 6-10):
    # Input:
    # 8888889888888888888888888
    # 8888881188282333888888888
    # 8888884111222323888888888
    # 8888881188282333888888888
    # 8888889888888888888888888

    # Extracting: cols 6-17 (12 cols):
    # 9888888888888 wait that's wrong.
    # Row 6: 8888889888888888888888888 -> col 6=9, rest=8.
    # Row 7: 8888881188282333888888888 -> col 6=1,7=1,8=8,9=8,10=2,11=8,12=2,13=3,14=3,15=3
    # Hmm: 88888|81188282333|88888888
    # So non-bg region: cols 6-15 (10 cols).

    # Shape 1:
    # 9888888888
    # 1188282333
    # 4111222323
    # 1188282333
    # 9888888888

    # Output cols 0-11:
    # 689888888888
    # 611188282333
    # 614111222323
    # 611188282333
    # 689888888888

    # So: output = shape 2 (3 cols: 689,611,614,611,689)
    #            + shape 1 (10 cols - 1: 888888888, 188282333, 111222323, 188282333, 888888888)
    # = 12 cols. But shape 2 is 5 cols wide, shape 1 is 10 cols wide.
    # 5 + 10 = 15, not 12.

    # So they overlap by 3 cols. The overlapping region: shape 2's right 3 cols = shape 1's left 3 cols.
    # Shape 2 right 3 cols: 988, 188, 488, 188, 988.
    # Shape 1 left 3 cols: 988, 118, 411, 118, 988.
    # These aren't the same, so it's not a simple overlay.

    # Actually, looking at the output: cols 0-4 = shape 2 (5 cols), cols 2-11 = shape 1 (10 cols).
    # Overlap at cols 2-4 (3 cols).
    # Shape 2 cols 2-4: 9,8,8 / 1,8,8 / 4,8,8 / 1,8,8 / 9,8,8.
    # Shape 1 cols 0-2: 9,8,8 / 1,1,8 / 4,1,1 / 1,1,8 / 9,8,8.
    # Output cols 2-4: 9,8,8 / 1,1,8 / 4,1,1 / 1,1,8 / 9,8,8.
    # So where both have non-bg values, take the non-bg. Where one is bg, take the other.

    # And shape 3 (3x3 at rows 18-20):
    # 333/323/333
    # In the output, it appears at cols 9-11: 333/323/333 (rows 0-4 right side):
    # Row 0: ...888 -> no 3s. Hmm.
    # Row 1: ...333. ✓
    # Row 2: ...323. ✓
    # Row 3: ...333. ✓
    # Row 4: ...888.

    # So shape 3 is placed at rows 1-3 (centered vertically), cols 9-11.
    # Shape 1's right 3 cols (cols 7-9 of shape 1): 888/333/323/333/888.
    # Wait, shape 1 cols 7-9:
    # Row 0: 8,8,8
    # Row 1: 2,3,3
    # Row 2: 2,3,2
    # Row 3: 2,3,3
    # Row 4: 8,8,8

    # And shape 3 placed there: 333/323/333 at rows 1-3.
    # Overlay: row 1 = max(232, 333) -> 333. But 232 vs 333 -> shape 3 wins (non-bg).
    # Actually shape 3 values are all non-bg (3s and 2), and shape 1 values at that position are also non-bg.
    # So the overlay prioritizes... shape 3?

    # Hmm, in the output cols 9-11:
    # Row 1: 333. Shape 1 at this position: 333 (cols 7-9 of shape 1, offset by 2 = cols 9-11 of output).
    # Wait I'm getting confused with indexing.

    # Let me just think of it as: all shapes are overlaid at specific positions,
    # with non-bg values taking priority (and if both are non-bg, one wins).

    # The key is figuring out WHERE each shape goes in the output.

    # I think the shapes share common "border" or "anchor" cells.
    # Shape 1 and shape 2 share the "9" cells (or some pattern).
    # Shape 1 and shape 3 share the "3" cells.

    # The "border color" of each shape acts as the "glue" - shapes stick together
    # at their matching borders.

    # For train 0:
    # Shape 2 border: 6 and 9. Interior: 1, 4, 8.
    # Shape 1 border: 9. Interior: 1, 4, 8, 2, 3.
    # Shape 3 border: 3. Interior: 2.

    # Shape 2's right side has 9s -> matches shape 1's left side (also 9).
    # Shape 1's right area has 3s -> matches shape 3 (all 3s border).

    # So the assembly is: shape 2 | shape 1 | shape 3,
    # where the matching borders overlap.

    # The 9-column of shape 2 (rightmost col of shape 2: 9,1,4,1,9 wait that has non-9s)
    # Actually shape 2 col 2: 9,1,4,1,9. The 9s are at rows 0,4.

    # I think the overlap is at the "border value" cells.
    # Shape 2 has 9s at its right edge (col 2 has 9s at rows 0,4).
    # Shape 1 has 9s at its left edge (col 0 has 9 at rows 0,4).
    # These align: shape 2's col 2 aligns with shape 1's col 0.

    # Shape 1 has 3s in its right area (row 1-3, cols 7-9: 333/323/333).
    # Shape 3 is entirely 3s and 2s.
    # Shape 1's 3-region overlaps with shape 3.

    # This overlap logic is complex to implement generically. Let me try a simpler approach:
    # Just overlay all shapes' bounding boxes onto the output,
    # aligning them so that matching border values overlap.

    # For now, let me try: find pairs of shapes that share a border value.
    # Align them by that border and combine.

    # This is too complex for a general solution. Let me skip.

    return grid  # placeholder


# Now let me add more solvers and test everything.

# ============================================================
# SOLVER: 64efde09
# Grid with "combs" (vertical/horizontal lines with teeth).
# Each comb has colored cells in its "slots".
# The output fills in the comb's slots based on the colored cells.
# ============================================================
def solve_64efde09(grid):
    R = len(grid)
    C = len(grid[0])
    bg = 8
    result = [row[:] for row in grid]

    # Looking at train 0:
    # There are vertical "combs" (lines of 4s and 2s with gaps).
    # Some rows adjacent to the comb have colored cells (5, 1, 6).
    # The output fills those colors along the comb's "arm" direction.

    # Comb 1 at cols 17-18: vertical line of 4s and 2s going from row 0 to row 6.
    # 44, 24, 23, 23, 23, 23, 24 then gap.
    # Color cells: row 4 col 16 = 5 (left of comb), row 5 col 16 = 5.
    # Row 5 col 0-6 has 84333344 -> 4 at col 1, 3s at cols 2-5.

    # In the output, the 5s from (4,16) extend leftward: row 4 cols 0-16 become 5s.
    # And the 1s from (5,15) extend leftward: row 5 cols 0-15 become 1s.

    # So the rule is: when a non-bg non-comb colored cell is adjacent to a comb,
    # it extends in the direction AWAY from the comb along that row/col
    # until it hits the grid edge or another comb.

    # Let me verify with train 0:
    # Row 4 (in output): 55555555555555555523888888888
    # 5s from col 0 to 16, then 23 at 17-18 (comb), then 8s.
    # Row 5 (input): 84333344888888888823888888888
    # Row 5 (output): 11111111111111111123888888888
    # 1s from col 0 to 16. Where does the 1 come from?
    # In the input, (25,0)=8, (25,7)=5, (25,8)=8. Hmm.
    # Row 25 (last data row): 88158868888881588688888888888.
    # I think the 1 and 5 come from the "color cells" adjacent to the comb.

    # Actually looking at the bottom-left: (25,7)=8. Row 25: 88858888888881888888888888888.
    # Wait, row 25: reading the actual input...
    # This is getting complex. Let me check the second comb.

    # Comb 2 at cols 6-7 (vertical):
    # Row 7: col 0 = 84333344. Wait let me read row 7:
    # Input row 7: 84333344888888888888888888888
    # So (7,0)=8, (7,1)=4, (7,2-5)=3333, (7,6)=4, (7,7)=4.
    # Comb at row 7: 4333344 -> this is a horizontal comb.

    # Actually, I think the "combs" are the long vertical lines (like 4-2-2-2-2-2-4).
    # These are like "rulers" with teeth. Each tooth position can have a color value.

    # Let me identify the combs:
    # A comb is a straight line of alternating 4s and 2s (or just 2s/4s).
    # The comb teeth point in a perpendicular direction.

    # In train 0, I see:
    # Vertical comb at cols 17-18: rows 0-6 have 44/24/23/23/23/23/24.
    # Another vertical comb at cols 18-19: rows 18-24 have similar.

    # Wait, there might be horizontal combs too.
    # Row 8: input: 82222224888888888808228808
    # That's 29 chars. Cols 0-6: 8222222, col 7: 4, rest: 888...
    # So cols 1-6 have 2s, col 7 has 4. This is horizontal.

    # I think the "combs" are the structures made of 4s and 2s.
    # The colored cells (3, 5, 6, 1, etc.) attached to the combs indicate
    # what color to fill in that direction.

    # In the output: where a colored cell is attached to a comb's "arm",
    # the color extends along the arm's direction to the grid edge.

    # The "arms" are the 2-cells that branch off from the main 4-spine.
    # Each arm row has a non-bg non-comb cell at its end (or start).

    # In train 0, comb at cols 17-18:
    # Row 0: 44 (top cap)
    # Row 1: 24 (2 at col 17, 4 at col 18)
    # Row 2: 23 (2 at col 17, 3 at col 18) <- 3 is a color
    # Row 3: 23 (same)
    # Row 4: 23
    # Row 5: 23
    # Row 6: 24 (bottom cap)

    # The 3 at col 18 rows 2-5: this is the "color" for the right arm.
    # In the output, these 3s extend rightward... but the output cols 18+
    # for rows 2-5: 23888888888. No extension.

    # Hmm. Let me look at specific changes from input to output:
    # For each cell that changed:
    # Row 1: col 0-16 changed? Input: 88888888888888888824888888888
    # Output: 66666666666666666624888888888. Cols 0-16 became 6.
    # Row 4: Input: 88888888888888888823888888888
    # Output: 55555555555555555523888888888. Cols 0-16 became 5.
    # Row 5: Input: 84333344888888888823888888888
    # Wait, (5,1)=4,(5,2-5)=3333,(5,6-7)=44.
    # Output: 11111111111111111123888888888. Cols 0-16 became 1.
    # But that replaced the 3333 and 44 with 1s!

    # So the fill overwrites existing content? Or maybe the 4/3 structure at row 5
    # is a different comb (horizontal) that gets consumed.

    # Looking at row 25 (last row):
    # Input: 88858888888881888888888888888
    # This is just data row, not transformed.

    # I think the rule is:
    # Each comb "arm" (the 2-cell extending from the 4-spine) has a color
    # attached at one end. That color fills the entire row (or column)
    # from the arm toward the opposite edge of the grid.

    # For the vertical comb at cols 17-18:
    # Row 1: arm extends left from col 17 (2 at col 17). Color at its end?
    # Row 2-5: 2 at col 17, 3 at col 18. The 3 is on the RIGHT side.
    # The fill goes LEFT: row 2-5 cols 0-16 get... what color?
    # Row 1 gets 6 (from where?), row 4 gets 5, row 5 gets 1.

    # Hmm, the colors 6, 5, 1 don't come from the comb itself.
    # Let me look for other colored cells in the grid.

    # Row 25: 88858888888881888888888888888
    # (25,3)=5, (25,13)=1. These are isolated colored cells.
    # Row 25 line: (25,0)=8,(25,1)=8,(25,2)=8,(25,3)=5,(25,4-12)=8s,(25,13)=1,(25,14-)=8s.

    # 5 at col 3 and 1 at col 13. These might indicate which row gets which color.

    # In the input, there are also other combs:
    # Horizontal comb at row 8: 82222224 (cols 1-7).
    # Vertical comb at cols 6-7: this has arms extending upward from row 7.

    # I think there are multiple combs and they interact. Each comb has a "key"
    # that maps its arms to colors. The key comes from the single colored cells.

    # This is getting extremely complex. Let me try a different approach:
    # Look for the "single colored cells" and figure out their relationship to the combs.

    # Actually, I recall from looking at the output that the colored cells
    # (5 at row 25 col 3, 1 at row 25 col 13, 6 at somewhere, etc.)
    # correspond to specific "arms" of the comb.

    # The comb at cols 17-18 has arms at rows 1-5.
    # The colored cell at (25,3)=5 is at the same position as some feature of another comb.
    # Maybe: each colored cell marks a row/col, and its value fills that row/col.

    # I think: find single isolated non-bg non-comb cells.
    # Each one determines the color for a specific row or column.
    # Then fill that row/column with the color, extending from the comb.

    # Let me try to find the "key" cells:
    # These are non-bg cells that are not part of any comb or shape.

    # For now, this is too complex. Skip.

    return grid  # placeholder


# ============================================================
# Additional solvers for remaining tasks
# ============================================================

def solve_5dbc8537(grid):
    R = len(grid)
    C = len(grid[0])

    # Grid is split into two halves by a separator.
    # Top half has a pattern of 9s (and a shape within).
    # Bottom half has shapes in rectangles.
    # Output = one of the halves, extracted.

    # Looking at train 0: 15x15 input -> 15x7 output.
    # The output is 15 rows, 7 cols. The input has a separator column of 9s at col 6.
    # Left half: cols 0-5 (6 cols of 9-pattern). Right half: cols 7-14 (8 cols).

    # Actually the 9s form a pattern in the left half, and the right half has
    # rectangular regions with colors.

    # Output is 15x7: the left 9-pattern extracted and formatted differently.

    # Hmm, the output is 15 rows x 7 cols:
    # 9888889
    # 9888889
    # 9933999
    # 9933999
    # 9933999
    # 9933999
    # 9933099
    # 9933999
    # 9119999
    # 9119999
    # 9955559
    # 9997799
    # 9997799
    # 9997799
    # 9955559

    # This has a 9-border (col 0 all 9s, col 6 all 9s).
    # Interior (cols 1-5): rows of colored cells.
    # The colors match those from the right half of the input.

    # I think: the left half has a "template" (pattern of 9s indicating positions),
    # and the right half has colored rectangles.
    # The output places the colored rectangles into the template positions.

    # This is complex. Skip for now.
    return grid  # placeholder


def solve_67e490f4(grid):
    R = len(grid)
    C = len(grid[0])
    bg = 1  # or detect

    # Large grid with mostly 1s, a rectangular region of 4s (bordered),
    # and scattered small shapes (2x2 or similar) of various colors.
    # Output = the 4-rectangle with the small shapes placed inside it
    # based on their values.

    # Looking at train 0:
    # 4-rectangle at rows 1-11, cols 14-24 (bordered by 4s).
    # Interior has a pattern of 4s and 1s.
    # Small shapes scattered around: 88 (2x2), 22 (2x2), 55 (1x2), 99 (2x2), etc.

    # Output is 11x11: the 4-rectangle.
    # But in the output, the small shapes' values are placed into the interior
    # of the 4-rectangle at positions where the rectangle has 1s (gaps).

    # This is a "lookup" task: each gap pattern in the 4-rectangle corresponds
    # to one of the small shapes, and its color fills that gap.

    # Complex. Skip for now.
    return grid  # placeholder


def solve_4a21e3da(grid):
    R = len(grid)
    C = len(grid[0])
    bg = 1
    result = [row[:] for row in grid]

    # Grid has a shape made of 7s on bg of 1s.
    # There's a single 2 cell somewhere.
    # In the output, the 7-shape gets "reflected" through the 2-cell,
    # creating a mirror image on the other side.

    # Looking at train 0:
    # 7-shape at rows 4-10, cols 4-12.
    # 2-cell at (0,8) and (8,17)... wait there are multiple 2s.
    # Actually: (0,8)=2, (8,17)=2.

    # In output:
    # (0,8)=2 stays. New content appears at rows 0-3 and cols 13-17.
    # The new content (77/77/17/17) looks like a reflection of the 7-shape.

    # The 2 at (0,8): this seems to be on the edge of the shape.
    # The shape has a "center" at the 2-cell, and it reflects across it.

    # In the output, row 8 (where the second 2 was):
    # Input: 111117111771111112
    # Output: 111117222772222222
    # So the 2 extends along the row from the 2-cell to the right edge!

    # And above the 2 at (0,8):
    # Input row 0: 111111112111111111
    # Output row 0: 771111112111111711
    # 7s appear at cols 0-1 and col 17.

    # This looks like the shape's boundary gets reflected through the 2-cell,
    # AND the 2 extends along the axis of reflection.

    # Actually, I think the 2 marks an axis of symmetry. The shape reflects
    # across a line passing through the 2-cell.

    # In train 0: 2 at (0,8). The axis goes through (0,8) vertically (col 8).
    # The shape at rows 4-10 gets reflected across col 8...
    # but the shape is already roughly symmetric around col 8.
    # The reflection creates new 7-cells on the opposite side.

    # Looking at (8,17)=2: this 2 is at the right edge of the shape.
    # In the output, 2s extend from (8,12) to (8,17) at row 8: 222222.
    # This is like the 2 "sliding" along the shape's edge.

    # Hmm, I think the rule is more nuanced.
    # The 2 indicates a direction. The shape "unfolds" in that direction,
    # creating a reflection on the other side of the 2-cell.

    # This is complex. Let me look at train 2 which might be simpler:
    # Shape at rows 7-11. 2 at (17,9).
    # Output: the shape reflects downward (toward row 17) through the 2.
    # New cells appear at rows 13-17.

    # So: the 2 is "below" the shape. The shape reflects downward through the 2.
    # The 2 creates a vertical axis at col 9, and the reflected shape appears
    # below the 2.

    # Hmm but the shape AND reflection would need to be separated by the 2.

    # Actually I think the 2 is the "endpoint" of a reflection axis.
    # The axis extends from the 2 through the shape's center.
    # The shape gets reflected across this axis to the other side of the 2.

    # In train 2: 2 at (17,9). Shape center around (9,9).
    # The shape needs to reflect downward from row 17.

    # Looking at train 2 output:
    # Row 12: 111111111211111111 -> 2 at col 9
    # Row 13: 117711111211111117 -> 7s at cols 2-3, col 17
    # Row 14: 177711111211111117 -> 7s at cols 1-3, col 17
    # Row 15: 771111111211111717 -> 7s at cols 0-1, cols 15,17
    # Row 16: 777711111211111777 -> 7s at cols 0-3, cols 15-17
    # Row 17: 177111111211111717 -> 7s at cols 1-3, cols 15,17

    # And the original shape (rows 7-11):
    # Row 7: 177111711111 -> 7 at col 1, 7 at col 6
    # Row 8: 777111711111 -> 7s at cols 0-2, 7 at col 6
    # Row 9: 771117171111 -> 7s at 0-1, 7s at 4,6
    # Row 10: 777777711111 -> 7s at 0-6
    # Row 11: 177177171111 -> 7 at 1, 7s at 3-4, 7 at 6

    # The reflected shape at rows 13-17 seems to be a mirror of rows 7-11
    # but reflected both horizontally and vertically?

    # Row 13 = mirror of row 11? Row 11: 177177171111 -> reflected: 111171771771.
    # Row 13 output: 117711111211111117 -> the 7-part: 77 at 2-3, 7 at 17.
    # Not a simple mirror.

    # This is very complex. I'll skip this.
    return grid  # placeholder


def solve_65b59efc(grid):
    # Complex mapping task. Skip.
    return grid


def solve_4c416de3(grid):
    R = len(grid)
    C = len(grid[0])
    bg = grid[0][0]
    result = [row[:] for row in grid]

    # Grid has rectangular boxes (bordered by 0s or similar).
    # Each box has a pattern inside. Some cells are special markers (not 0 or bg).
    # In the output, the special markers get "reflected" outside the box.

    # Looking at train 0:
    # Box 1 at rows 3-10, cols 3-10: bordered by 0s.
    # Interior: rows 4-9, cols 4-9. Contains bg (1), 3, 8, 4, 2.
    # Special cells: 3 at (5,5), 8 at (5,7-8), 4 at (8,5), 2 at (8,8).

    # In the output:
    # 3 at (5,5): reflected to... (2,4) and (2,3) become 3.
    # 4 at (8,5): reflected to (11,4) and (11,3) become 4.
    # 8 at (5,7-8): reflected to (2,9-10) become 8?
    # 2 at (8,8): reflected to (9,9-10) become 2?

    # Actually, looking at the output more carefully:
    # Row 2: 113311111188111111111 -> 3s at cols 2-3, 8s at cols 9-10.
    # Row 11: 114411111122111111111 -> 4s at cols 2-3, 2s at cols 9-10.
    # These appear OUTSIDE the box.

    # The special cells are at box corners (approximately):
    # 3 at (5,5): upper-left of interior
    # 8 at (5,7-8) and row 2: upper-right
    # 4 at (8,5): lower-left
    # 2 at (8,8): lower-right

    # The reflections go OUTSIDE the box:
    # 3 appears at row 2 cols 2-3 (above-left of box)
    # 8 appears at row 2 cols 9-10 (above-right of box)
    # 4 appears at row 11 cols 2-3 (below-left of box)
    # 2 appears at row 11 cols 9-10 (below-right of box)

    # Each corner marker reflects diagonally outside the box.
    # The distance from the corner to the marker = distance from corner to reflected position.

    # In the box: top-left corner at (3,3), marker 3 at (5,5).
    # Offset: (2,2) from top-left.
    # Reflection: go (-2,-2) from top-left: (1,1). But the 3s appear at (2,2-3).

    # Hmm, not exactly diagonal. Let me look more carefully.

    # Actually, looking at the box corner structure:
    # Box has 0-border. The corners of the 0-border are:
    # Top-left: (3,3), Top-right: (3,10)
    # Bottom-left: (10,3), Bottom-right: (10,10)

    # Wait, the box extends from row 3-10, col 3-10? Let me check.
    # Row 3: 111000000088111111111 -> 0s at cols 3-9, 8s at 10-11.
    # Hmm, 0s at 3-9 and 88 at 10-11. So the box might extend to col 11.
    # Row 10: 111000000001111111111 -> 0s at cols 3-11.
    # Actually row 10: 114400000022111111111 in output. Let me re-read input.
    # Input row 10: 111000000001111111111 -> 0s at cols 3-11.

    # So the box is rows 3-10, cols 3-11 (9 cols).
    # 0-border: outer ring of 0s.
    # Interior: rows 4-9, cols 4-10 (7x7).

    # Interior markers:
    # (5,5): 3, (5,7): 1 (bg), (5,8): 8
    # Hmm, input row 5: 111013118101111111111
    # Cols: 3=0, 4=1, 5=3, 6=1, 7=1, 8=8, 9=1, 10=0
    # So marker 3 at col 5, marker 8 at col 8.

    # (8,5): 4, (8,8): 2
    # Input row 8: 111014112101111111111
    # Cols: 3=0, 4=1, 5=4, 6=1, 7=1, 8=2, 9=1, 10=0

    # So markers at interior positions:
    # (5,5)=3, (5,8)=8, (8,5)=4, (8,8)=2

    # Box corners (interior): (4,4) top-left, (4,10) top-right, (9,4) bottom-left, (9,10) bottom-right.
    # Actually interior corners: (4,4), (4,10), (9,4), (9,10).

    # Marker 3 at (5,5): offset (1,1) from top-left interior corner (4,4).
    # Marker 8 at (5,8): offset (1,4) from top-left, or (1,-2) from top-right (4,10).
    # Marker 4 at (8,5): offset (4,1) from top-left, or (-1,1) from bottom-left (9,4).
    # Marker 2 at (8,8): offset (-1,-2) from bottom-right (9,10).

    # In output:
    # 3 appears at (2,2-3): this is at row 2, cols 2-3.
    # Relative to box's top-left corner (3,3): offset (-1,-1) and (-1,0).

    # The marker 3 is at offset (2,2) from box top-left corner.
    # The reflection goes to (-2,-2)... but the output 3 is at (2,2-3).
    # Hmm, offset from box corner (3,3): (2-3, 2-3) = (-1,-1) and (2-3, 3-3) = (-1,0).

    # That's a 1x2 block of 3s. The marker was a single cell.
    # So the reflection creates a block?

    # Looking at output train 1 for more examples...
    # This is getting very complex. Skip.

    return grid


# ============================================================
# Main: test all solvers and generate solutions
# ============================================================
def main():
    solvers = {
        '78332cb0': solve_78332cb0,
        '7491f3cf': solve_7491f3cf_v2,
        '45a5af55': solve_45a5af55,
        '5545f144': solve_5545f144,
        '7b5033c1': solve_7b5033c1,
        '53fb4810': solve_53fb4810,
        '581f7754': solve_581f7754,
        '58490d8a': solve_58490d8a,
        '7b3084d4': solve_7b3084d4,
        '6ffbe589': solve_6ffbe589,
        '71e489b6': solve_71e489b6,
        '6e453dd6': solve_6e453dd6_NEW,
        '4c7dc4dd': solve_4c7dc4dd,
        '4a21e3da': solve_4a21e3da,
        '409aa875': solve_409aa875,
        '446ef5d2': solve_446ef5d2,
        '4c3d4a41': solve_4c3d4a41,
        '4c416de3': solve_4c416de3,
        '4e34c42c': solve_4e34c42c,
        '5961cc34': solve_5961cc34,
        '5dbc8537': solve_5dbc8537,
        '58f5dbd5': solve_58f5dbd5,
        '62593bfd': solve_62593bfd,
        '64efde09': solve_64efde09,
        '65b59efc': solve_65b59efc,
        '67e490f4': solve_67e490f4,
        '6e4f6532': solve_6e4f6532,
        '7b0280bc': solve_7b0280bc,
        '7666fa5d': solve_7666fa5d,
        '3e6067c3': lambda grid: grid,  # placeholder
    }

    solutions = {}
    passed = []
    failed = []

    for tid in TASK_IDS:
        solver = solvers.get(tid)
        if solver is None:
            print(f"{tid}: NO SOLVER")
            failed.append(tid)
            continue

        try:
            if test_solver(tid, solver):
                print(f"{tid}: PASS")
                passed.append(tid)
                results = apply_solver(tid, solver)
                solutions[tid] = results
            else:
                print(f"{tid}: FAIL (training mismatch)")
                failed.append(tid)
        except Exception as e:
            print(f"{tid}: ERROR {e}")
            failed.append(tid)

    print(f"\nPassed: {len(passed)}/{len(TASK_IDS)}")
    print(f"Failed: {len(failed)}")

    # Save solutions
    with open(OUT_FILE, 'w') as f:
        json.dump(solutions, f)
    print(f"Saved {len(solutions)} solutions to {OUT_FILE}")

if __name__ == '__main__':
    main()
