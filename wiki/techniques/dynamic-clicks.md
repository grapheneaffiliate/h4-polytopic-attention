# Dynamic Click Refresh

## Discovery
`_get_valid_clickable_actions()` returns DIFFERENT click targets depending on game state. Must call it at EACH BFS node, not just at the start.

## Impact
Solved 5 games that were previously stuck: AR25, CN04, KA59, M0R0, SK48.

## Why It Works
Many games have interactive sprites that appear/disappear based on state. The sys_click targets change as:
- Objects move to new positions
- New interactive elements become visible
- Completed items remove their click targets

## Implementation
```python
for a in all_actions:
    c = copy.deepcopy(g)
    step_game(c, a)
    # ... check win ...
    
    # CRITICAL: refresh clicks from the COPIED state
    try:
        dyn = g._get_valid_clickable_actions()
        all_actions_next = list(kbd) + list(dyn)
    except:
        pass
```

## Games Where It Matters
- **AR25**: Click targets change as objects are selected/moved
- **CN04**: Shape pieces become clickable after movement
- **KA59**: Numbered tiles activate/deactivate
- **M0R0**: Object selection changes available clicks
- **SK48**: State-dependent interactive elements
