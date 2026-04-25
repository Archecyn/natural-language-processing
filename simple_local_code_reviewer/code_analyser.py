"""
code_analyser.py
─────────────────
Builds structured prompts for the LLM, sends them via LLMClient, and
parses the structured response back into (new_content, summary) tuples.
"""

import logging
import re
from typing import Optional

from llm_client import LLMClient

log = logging.getLogger(__name__)

# ── System prompts ─────────────────────────────────────────────────────────────

_SYSTEM_ERRORS = """\
You are an expert code reviewer specialising in bug detection and correctness.
Your task is to identify and fix real errors in the provided source file.

Focus on:
- Syntax errors and typos that would cause runtime failures
- Logic bugs (off-by-one, wrong comparisons, incorrect boolean logic)
- Unhandled exceptions or missing error handling
- Null/undefined dereferences and type mismatches
- Resource leaks (unclosed files, connections)
- Security vulnerabilities (SQL injection, path traversal, hardcoded secrets)
- Incorrect API usage or deprecated patterns

Rules:
- Only change code that contains real errors. Do NOT refactor working code.
- Preserve the original structure, variable names, and style where possible.
- Never remove intentional TODO/FIXME comments.
"""

_SYSTEM_OPTIMISE = """\
You are an expert software engineer specialising in code quality and performance.
Your task is to improve the provided source file for performance and readability.

Focus on:
- Algorithmic improvements (e.g. O(n²) → O(n log n), remove redundant loops)
- Pythonic / idiomatic rewrites (list comprehensions, context managers, etc.)
- **DUPLICATE CODE ELIMINATION**: Extract repeated code blocks into functions, variables, or helper methods
- Removing dead code, duplicate logic, and unnecessary variables
- Improving naming: variables, functions, and parameters should be self-documenting
- Adding or improving docstrings and inline comments for complex logic
- Breaking overly long functions into smaller, focused ones
- Applying language-specific best practices

Rules:
- Maintain 100% functional equivalence — do not change observable behaviour.
- Keep the same public API (function signatures, exports, class interfaces).
- Do not introduce new dependencies.
"""

_SYSTEM_BOTH = """\
You are an expert code reviewer and software engineer.
Your task is to both fix errors AND optimise the provided source file.

Fix errors:
- Syntax/runtime errors, logic bugs, security issues, resource leaks

Optimise:
- Performance improvements, idiomatic rewrites, better naming, docstrings
- **DUPLICATE CODE ELIMINATION**: Extract repeated code blocks into functions, variables, or helper methods

Rules:
- Maintain 100% functional equivalence for working code.
- Keep the same public API.
- Do not introduce new dependencies.
"""

_SYSTEMS = {
    "errors": _SYSTEM_ERRORS,
    "optimise": _SYSTEM_OPTIMISE,
    "both": _SYSTEM_BOTH,
}

# ── User prompt template ───────────────────────────────────────────────────────

_USER_TEMPLATE = """\
File: {path}

<original_code>
{code}
</original_code>

Instructions:
1. Analyse the code above for errors and optimization opportunities.
2. Pay special attention to DUPLICATE CODE - if you see the same code repeated, extract it into a function or variable.
3. If no changes are needed, reply with exactly: NO_CHANGES_NEEDED
4. Otherwise, respond in this EXACT format (do not deviate):

SUMMARY:
<A concise bullet-point list of every change made and why>

REVISED_CODE:
```
<the complete revised file — no truncation, no placeholders>
```
"""

# ── Analyser class ─────────────────────────────────────────────────────────────

class CodeAnalyser:
    def __init__(
        self,
        llm: LLMClient,
        mode: str = "both",
    ):
        if mode not in _SYSTEMS:
            raise ValueError(f"mode must be one of {list(_SYSTEMS)}, got '{mode}'")
        self.llm = llm
        self.mode = mode
        self.system_prompt = _SYSTEMS[mode]

    def analyse(
        self,
        file_path: str,
        original_code: str,
    ) -> Optional[tuple[str, str]]:
        """
        Run the LLM analysis on a single file.

        Returns:
            (revised_code, summary)  if changes are suggested
            None                     if no changes are needed or parsing fails
        """
        # Truncate very large files to avoid exceeding context
        code = self._truncate(original_code, max_chars=12_000)

        user_prompt = _USER_TEMPLATE.format(path=file_path, code=code)

        try:
            raw = self.llm.chat(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            log.warning(f"LLM inference failed for {file_path}: {exc}")
            return None

        # Log LLM response for debugging (first 300 chars)
        log.debug(f"LLM response for {file_path}: {raw[:300]}...")

        return self._parse(raw, original_code)

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _truncate(code: str, max_chars: int) -> str:
        if len(code) <= max_chars:
            return code
        half = max_chars // 2
        truncation_note = "\n\n# ... [file truncated for context length] ...\n\n"
        return code[:half] + truncation_note + code[-half:]

    @staticmethod
    def _parse(raw: str, original_code: str) -> Optional[tuple[str, str]]:
        """Parse the structured LLM response into (code, summary)."""

        # Check for no changes response
        if "NO_CHANGES_NEEDED" in raw.upper():
            log.debug("LLM indicated no changes needed")
            return None

        # Extract summary - look for SUMMARY: followed by REVISED_CODE: or end
        summary_match = re.search(
            r"SUMMARY:\s*(.*?)(?=REVISED_CODE:|$)", raw, re.DOTALL | re.IGNORECASE
        )
        summary = summary_match.group(1).strip() if summary_match else "Changes applied."

        # Extract code block - look for REVISED_CODE: followed by ```
        code_match = re.search(
            r"REVISED_CODE:\s*```[^\n]*\n(.*?)\n```",
            raw,
            re.DOTALL | re.IGNORECASE,
        )
        if not code_match:
            # Fallback: look for any triple-backtick block after REVISED_CODE
            fallback_match = re.search(
                r"REVISED_CODE:.*?(```[^\n]*\n.*?```)",
                raw,
                re.DOTALL | re.IGNORECASE,
            )
            if fallback_match:
                code_match = re.search(r"```[^\n]*\n(.*?)\n```", fallback_match.group(1), re.DOTALL)
            else:
                # Last resort: any triple-backtick block
                code_match = re.search(r"```[^\n]*\n(.*?)\n```", raw, re.DOTALL)

        if not code_match:
            log.warning("Could not extract code block from LLM response — skipping file.")
            log.debug(f"Raw LLM output:\n{raw}")
            return None

        revised = code_match.group(1)

        # Sanity check: don't return an empty or too-short file
        if len(revised.strip()) < 10:
            log.warning("Extracted code is suspiciously short — skipping.")
            return None

        return revised, summary
