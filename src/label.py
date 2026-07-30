"""Mine preference pairs from failures already recorded in the transcripts.

Two naturally-occurring negative signals, no labeling required:
  error  -> the tool returned is_error, and a later call to the same tool worked
  interrupt -> the user cut the action off, and a later call to the same tool stood

The rejected action is what the expert tried first; the chosen action is what it
did instead. Identical retries are dropped -- a command rerun verbatim after a
flaky failure carries no preference signal.

Reads data/actions.jsonl, writes data/pairs.jsonl.
"""

import argparse
import collections
import json
import os
import random
import re

HERE = os.path.dirname(__file__)
IN = os.path.join(HERE, "..", "data", "actions.jsonl")
OUT = os.path.join(HERE, "..", "data", "pairs.jsonl")

WINDOW = 3  # how many subsequent actions may count as the retry

# Measured precision on a 27-pair hand review: ~20 genuine fixes, 3 topic pivots
# where the "retry" is simply different work (~11% noise). A token-Jaccard
# similarity filter was tried and rejected: the worst pivot scored 0.129 while
# three genuine fixes scored 0.000-0.051, so every threshold that caught the
# pivots destroyed more real corrections than it saved. Left unfiltered on
# purpose -- DPO tolerates this noise level.
# Tools whose failures are informative. TodoWrite/StructuredOutput failures are
# schema noise, not procedural knowledge.
PAIR_TOOLS = {"Bash", "Edit", "Write", "Read", "WebFetch", "Glob", "Grep", "NotebookEdit"}


def norm(text: str) -> str:
    """Whitespace-insensitive form, for deciding whether a retry actually differs."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def action_key(action: dict) -> str:
    """The part of an action a retry would meaningfully change."""
    tool, inp = action.get("tool"), action.get("input", {})
    if not isinstance(inp, dict):
        return norm(str(inp))
    if tool == "Bash":
        return norm(inp.get("command", ""))
    if tool in ("Edit", "NotebookEdit"):
        return norm(f"{inp.get('file_path', '')}||{inp.get('old_string', '')}")
    if tool == "Write":
        return norm(f"{inp.get('file_path', '')}||{inp.get('content', '')[:500]}")
    if tool == "Read":
        return norm(inp.get("file_path", ""))
    return norm(json.dumps(inp, sort_keys=True, default=str))


def load_actions(path):
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def mine(rows):
    """rows: actions of one session in transcript order."""
    for i, bad in enumerate(rows):
        out = bad["outcome"]
        reason = "error" if out.get("is_error") else ("interrupt" if out.get("interrupted") else None)
        if not reason:
            continue
        tool = bad["action"].get("tool")
        if tool not in PAIR_TOOLS:
            continue
        bad_key = action_key(bad["action"])
        for good in rows[i + 1 : i + 1 + WINDOW]:
            if good["action"].get("tool") != tool:
                continue
            gout = good["outcome"]
            if gout.get("is_error") or gout.get("interrupted"):
                continue
            good_key = action_key(good["action"])
            if not good_key or good_key == bad_key:
                break  # verbatim rerun of a flaky call: no preference signal
            yield {
                "project": bad["project"],
                "tier": bad["tier"],
                "src": bad["src"],
                "session": bad["session"],
                "ts": bad["ts"],
                "cwd": bad["cwd"],
                "gitBranch": bad.get("gitBranch"),
                "reason": reason,
                "tool": tool,
                "context": bad["context"],
                "rejected": bad["action"],
                "chosen": good["action"],
            }
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=IN)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sample", type=int, default=0, help="print N random pairs for review")
    args = ap.parse_args()

    # Group by source transcript, preserving order. Grouping by sessionId would
    # merge every subagent of a session into one stream and mint false pairs
    # across unrelated agents at the file boundaries.
    sessions = collections.OrderedDict()
    total = 0
    for row in load_actions(args.inp):
        total += 1
        sessions.setdefault(row["src"], []).append(row)

    pairs = []
    for rows in sessions.values():
        pairs.extend(mine(rows))

    with open(args.out, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p, default=str) + "\n")

    by_tool = collections.Counter(p["tool"] for p in pairs)
    by_reason = collections.Counter(p["reason"] for p in pairs)
    by_proj = collections.Counter(p["project"] for p in pairs)
    print(f"actions scanned : {total:,}")
    print(f"sessions        : {len(sessions):,}")
    print(f"PAIRS MINED     : {len(pairs):,}")
    print(f"  by reason: {dict(by_reason)}")
    print(f"  by tool  : {dict(by_tool.most_common())}")
    print(f"  by project:")
    for k, v in by_proj.most_common():
        print(f"    {k:<45}{v:>6}")
    print(f"\nwrote {args.out}")

    if args.sample:
        random.seed(1)
        print(f"\n{'=' * 78}\nRANDOM SAMPLE FOR MANUAL REVIEW (n={args.sample})\n{'=' * 78}")
        for k, p in enumerate(random.sample(pairs, min(args.sample, len(pairs))), 1):
            rej = action_key(p["rejected"])[:340]
            cho = action_key(p["chosen"])[:340]
            print(f"\n--- {k}. [{p['tool']}/{p['reason']}] {p['project']}")
            print(f"  REJECTED: {rej}")
            print(f"  CHOSEN  : {cho}")


if __name__ == "__main__":
    main()
