"""Parser sanity checks. A broken parser reports 0% accuracy and looks exactly
like a failed experiment, so these run before trusting any generation number."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_eval import parse_action, cmd_of, f1  # noqa: E402

CASES = [
    ('{"tool": "Bash", "input": {"command": "pnpm vitest run"}}', "Bash", "pnpm vitest run"),
    ('```json\n{"tool":"Read","input":{"file_path":"/a/b.ts"}}\n```', "Read", "/a/b.ts"),
    ('Sure! {"tool": "Edit", "input": {"file_path": "x.ts", "old_string": "a"}}', "Edit", "x.ts"),
    ('{"tool": "Bash", "input": {"command": "bun test src/eng', "Bash", ""),   # truncated
    ("no json here at all", "None", ""),
    ('{"tool": "TodoWrite", "input": {"todos": []}}', "TodoWrite", ""),
]


def main():
    bad = 0
    for text, want_tool, want_cmd in CASES:
        act = parse_action(text)
        got_tool, got_cmd = str(act["tool"]), cmd_of(act)
        ok = got_tool == want_tool and got_cmd == want_cmd
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  tool={got_tool!r:<12} cmd={got_cmd[:32]!r}")

    print()
    checks = [
        ("bun vs pnpm (should be partial)", f1("bun test src/engine", "pnpm vitest run src/engine"), 0.05, 0.75),
        ("identical (should be 1.0)", f1("pnpm vitest run", "pnpm vitest run"), 0.99, 1.01),
        ("disjoint (should be 0.0)", f1("git status", "npm install"), -0.01, 0.01),
    ]
    for label, val, lo, hi in checks:
        ok = lo <= val <= hi
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<34} {val:.3f}")

    print(f"\n  {'ALL PASS' if not bad else f'{bad} FAILURES'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
