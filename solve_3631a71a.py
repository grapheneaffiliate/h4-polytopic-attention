import json

def solve(inp):
    n = len(inp)
    out = [row[:] for row in inp]
    def equivalents(r, c):
        positions = set()
        for rr, cc in [(r,c), (c,r), (31-r,c), (r,31-c), (31-r,31-c), (c,31-r), (31-c,r), (31-c,31-r)]:
            if 0 <= rr < n and 0 <= cc < n:
                positions.add((rr, cc))
        return positions
    changed = True
    while changed:
        changed = False
        for r in range(n):
            for c in range(n):
                if out[r][c] == 9:
                    for rr, cc in equivalents(r, c):
                        if out[rr][cc] != 9:
                            out[r][c] = out[rr][cc]
                            changed = True
                            break
    return out

if __name__ == '__main__':
    with open('data/arc1/3631a71a.json') as f:
        data = json.load(f)

    # Verify on all training pairs
    for i, pair in enumerate(data['train']):
        result = solve(pair['input'])
        if result == pair['output']:
            print(f'Train {i}: PASS')
        else:
            diffs = sum(1 for r in range(30) for c in range(30) if result[r][c] != pair['output'][r][c])
            print(f'Train {i}: FAIL ({diffs} diffs)')

    # Solve test
    test_result = solve(data['test'][0]['input'])
    expected = data['test'][0]['output']
    if test_result == expected:
        print('Test: PASS')
    else:
        diffs = sum(1 for r in range(30) for c in range(30) if test_result[r][c] != expected[r][c])
        print(f'Test: FAIL ({diffs} diffs)')

    print('\nTest answer:')
    print(json.dumps(test_result))
