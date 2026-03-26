"""
Code Verifier — Sprint Contract Pattern

Generate code → Execute → Verify → Fix if needed.

The code specialist generates Python. We run it in a sandbox.
If it crashes or produces wrong output, we feed the error back
for a second attempt with the traceback as context.

This is the "evaluator" from the Anthropic harness pattern:
planner generates, evaluator checks, generator fixes.
"""

import re
import subprocess
import sys
import tempfile
import os


def extract_python_code(text: str) -> str:
    """Extract Python code from markdown code blocks or raw text."""
    # Try fenced code blocks first
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        return "\n\n".join(blocks)

    # Try indented blocks (lines starting with spaces after a colon)
    lines = text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        # Heuristic: looks like Python code
        if any(stripped.startswith(kw) for kw in
               ['def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ',
                'return ', 'print(', 'try:', 'except', 'with ', '#']):
            in_code = True
        if in_code:
            if stripped == '' and code_lines and code_lines[-1].strip() == '':
                in_code = False
                continue
            code_lines.append(line)

    return "\n".join(code_lines) if code_lines else ""


def run_python_code(code: str, timeout: int = 10) -> dict:
    """Execute Python code in a subprocess sandbox.

    Returns dict with:
        success: bool
        stdout: str
        stderr: str
        returncode: int
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip(),
            'returncode': result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Execution timed out after {timeout}s',
            'returncode': -1,
        }
    finally:
        os.unlink(tmp_path)


def verify_code_response(response_text: str) -> dict:
    """Extract code from a specialist response, run it, return results.

    Returns dict with:
        has_code: bool
        code: str
        execution: dict (from run_python_code) or None
        verified: bool (True if code ran without errors)
    """
    code = extract_python_code(response_text)
    if not code:
        return {'has_code': False, 'code': '', 'execution': None, 'verified': False}

    execution = run_python_code(code)
    return {
        'has_code': True,
        'code': code,
        'execution': execution,
        'verified': execution['success'],
    }


def check_output_properties(query: str, stdout: str) -> dict:
    """Check mathematical properties of code output.

    Doesn't need a reference answer — just verifies the output
    satisfies the properties implied by the query.

    Returns dict with:
        checked: bool (True if we could check something)
        passed: bool
        failures: list of str (property violations)
    """
    failures = []
    q_lower = query.lower()
    output = stdout.strip()

    if not output:
        return {'checked': False, 'passed': True, 'failures': []}

    # Try to parse output as a Python list
    parsed_list = None
    try:
        import ast
        parsed_list = ast.literal_eval(output)
        if not isinstance(parsed_list, list):
            parsed_list = None
    except Exception:
        pass

    # Try to extract input list from query
    input_list = None
    import re
    list_match = re.search(r'\[[\d,\s\-]+\]', query)
    if list_match:
        try:
            import ast
            input_list = ast.literal_eval(list_match.group())
        except Exception:
            pass

    if parsed_list is None or input_list is None:
        return {'checked': False, 'passed': True, 'failures': []}

    # Property checks for subsequence-type problems
    if any(kw in q_lower for kw in ['subsequence', 'subarray', 'subset']):

        # Check: output is a subsequence of input (elements appear in order)
        if 'subsequence' in q_lower:
            it = iter(input_list)
            is_subseq = all(elem in it for elem in parsed_list)
            # More careful check: elements appear in input in order
            idx = 0
            is_subseq = True
            for elem in parsed_list:
                found = False
                while idx < len(input_list):
                    if input_list[idx] == elem:
                        idx += 1
                        found = True
                        break
                    idx += 1
                if not found:
                    is_subseq = False
                    break
            if not is_subseq:
                failures.append(f"output {parsed_list} is not a subsequence of input {input_list}")

        # Check: strictly increasing
        if 'increasing' in q_lower:
            for i in range(len(parsed_list) - 1):
                if parsed_list[i] >= parsed_list[i + 1]:
                    failures.append(
                        f"not strictly increasing: {parsed_list[i]} >= {parsed_list[i+1]} "
                        f"at positions {i},{i+1} in {parsed_list}"
                    )
                    break

        # Check: strictly decreasing
        if 'decreasing' in q_lower:
            for i in range(len(parsed_list) - 1):
                if parsed_list[i] <= parsed_list[i + 1]:
                    failures.append(
                        f"not strictly decreasing: {parsed_list[i]} <= {parsed_list[i+1]}"
                    )
                    break

    # Property checks for sorting
    if any(kw in q_lower for kw in ['sort', 'sorted', 'order']):
        if 'descend' in q_lower:
            for i in range(len(parsed_list) - 1):
                if parsed_list[i] < parsed_list[i + 1]:
                    failures.append(f"not sorted descending: {parsed_list[i]} < {parsed_list[i+1]}")
                    break
        else:
            for i in range(len(parsed_list) - 1):
                if parsed_list[i] > parsed_list[i + 1]:
                    failures.append(f"not sorted ascending: {parsed_list[i]} > {parsed_list[i+1]}")
                    break

        # Check: same elements as input (permutation)
        if input_list and sorted(parsed_list) != sorted(input_list):
            failures.append(f"output elements {sorted(parsed_list)} != input elements {sorted(input_list)}")

    # Property checks for search/find
    if 'max' in q_lower or 'longest' in q_lower or 'largest' in q_lower:
        # We can't verify optimality without a reference, but we can flag empty results
        if len(parsed_list) == 0:
            failures.append("output is empty for a max/longest query")

    checked = len(failures) > 0 or (parsed_list is not None and input_list is not None)
    return {
        'checked': checked,
        'passed': len(failures) == 0,
        'failures': failures,
    }


def build_fix_prompt(query: str, code: str, error: str) -> str:
    """Build a prompt asking the specialist to fix the code."""
    return (
        f"The following code was written to answer: {query}\n\n"
        f"```python\n{code}\n```\n\n"
        f"But it produced this error:\n```\n{error}\n```\n\n"
        f"Fix the code. Show only the corrected version."
    )
