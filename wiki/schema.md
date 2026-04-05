# Wiki Schema

## Conventions for LLM sessions maintaining this wiki

### Game Pages (`wiki/games/<game_id>.md`)
Every game page should have:
- **Status line**: `X/Y solved | Tags: ... | Baselines: ...`
- **Mechanic**: one sentence
- **Win Condition**: exact function/check from source
- **Why Unsolved** (if applicable): specific reason with evidence
- **What's Needed**: concrete next step
- **Per-Level Notes**: table with level-specific details
- **Key Variables**: obfuscated names decoded
- **Source**: path to game file

### Technique Pages (`wiki/techniques/<name>.md`)
- **Concept**: what it is
- **Impact**: which games it helped
- **Implementation**: key code pattern
- **Pitfalls**: what goes wrong

### Updating Rules
1. When a game is solved/extended: update its page AND `index.md` AND `overview.md`
2. When a new technique is discovered: create a technique page AND link from relevant game pages
3. When a bug is found: add to the game page's notes AND to `techniques/deepcopy-bug.md` if relevant
4. Always update `log.md` with date and summary

### File Naming
- Game pages: lowercase game ID (e.g., `lp85.md`)
- Technique pages: kebab-case (e.g., `abstract-bfs.md`)
- Use `[[links]]` for Obsidian compatibility
- YAML frontmatter optional (for Dataview queries)
