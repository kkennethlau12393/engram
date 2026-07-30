"""Turn extracted actions into MLX-LM prompt/completion datasets.

Split policy: per-repo temporal, assigned at TRANSCRIPT level. Consecutive
actions inside one session are near-duplicates, so splitting on individual
actions would leak a session across train and test. Whole transcripts move
together, ordered by their first timestamp, so test is always the most recent
slice of each repo's own history -- which is what deployment actually looks like.

Writes:
  data/sft/pooled/{train,valid}.jsonl      shared adapter
  data/sft/<repo>/{train,valid}.jsonl      per-repo adapters
  data/test/<repo>.jsonl                   held-out eval
  data/dpo/{train,valid}.jsonl             preference pairs, same split boundary
"""

import argparse
import collections
import json
import os

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")

# Rough chars-per-token for Qwen; used to budget the prompt without paying for
# tokenization of 109K records.
CPT = 3.6
MAX_PROMPT_TOKENS = 1600
MAX_PROMPT_CHARS = int(MAX_PROMPT_TOKENS * CPT)
MAX_COMPLETION_CHARS = 1400

TEST_FRAC = 0.15
VALID_FRAC = 0.05

# StructuredOutput is a harness artifact: its "input" is a workflow agent's
# return payload, not an action taken in a repo. Excluded from next-action data.
DROP_TOOLS = {"StructuredOutput"}

SYSTEM = (
    "You predict the next tool call a senior engineer would make in this repository. "
    "Reply with one JSON object: {\"tool\": ..., \"input\": {...}}. No prose."
)


# Claude Code encodes a project dir as its cwd with "/" -> "-", so the home
# prefix is derivable rather than hardcoded to one machine's username.
HOME_TAG = os.path.expanduser("~").replace("/", "-")


def short_project(p):
    """-Users-<you>-Desktop-myrepo -> myrepo ; -Users-<you> -> home"""
    name = p.replace(HOME_TAG, "", 1).lstrip("-")
    for prefix in ("Desktop-", "Documents-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name or "home"


def build_prompt(row):
    """Newest context first into the budget, then re-emit oldest-first."""
    head = [f"cwd: {row.get('cwd')}"]
    if row.get("gitBranch"):
        head.append(f"branch: {row['gitBranch']}")
    head.append("recent:")
    head_txt = "\n".join(head)

    budget = MAX_PROMPT_CHARS - len(head_txt) - 24
    kept = []
    for line in reversed(row.get("context") or []):
        if budget <= 0:
            break
        if len(line) > budget:
            line = line[: max(0, budget - 16)] + "…"
        kept.append(line)
        budget -= len(line) + 1
    kept.reverse()
    return f"{head_txt}\n" + "\n".join(kept) + "\nNEXT_ACTION:"


def build_completion(action):
    txt = json.dumps(
        {"tool": action.get("tool"), "input": action.get("input", {})},
        default=str,
        ensure_ascii=False,
    )
    return txt[:MAX_COMPLETION_CHARS]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", default=os.path.join(DATA, "actions.jsonl"))
    ap.add_argument("--pairs", default=os.path.join(DATA, "pairs.jsonl"))
    ap.add_argument("--min-repo-actions", type=int, default=2000,
                    help="repos below this get pooled into the shared adapter only")
    args = ap.parse_args()

    # Pass 1: transcript -> (project, first_ts, rows)
    transcripts = collections.OrderedDict()
    dropped = 0
    with open(args.actions, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["action"].get("tool") in DROP_TOOLS:
                dropped += 1
                continue
            t = transcripts.setdefault(row["src"], {"project": row["project"], "ts": None, "rows": []})
            t["rows"].append(row)
            ts = row.get("ts")
            if ts and (t["ts"] is None or ts < t["ts"]):
                t["ts"] = ts

    # Pass 2: per-repo temporal assignment at transcript level
    by_project = collections.defaultdict(list)
    for src, t in transcripts.items():
        by_project[t["project"]].append((t["ts"] or "", src))

    split_of = {}
    for proj, items in by_project.items():
        items.sort()
        n = len(items)
        n_test = max(1, int(n * TEST_FRAC))
        n_valid = max(1, int(n * VALID_FRAC))
        n_train = max(0, n - n_test - n_valid)
        for i, (_, src) in enumerate(items):
            split_of[src] = "train" if i < n_train else ("valid" if i < n_train + n_valid else "test")

    # Pass 3: emit
    buckets = collections.defaultdict(list)          # (proj, split) -> rows
    counts = collections.Counter()
    repo_totals = collections.Counter()

    for src, t in transcripts.items():
        split = split_of[src]
        proj = short_project(t["project"])
        repo_totals[proj] += len(t["rows"])
        for row in t["rows"]:
            out = row["outcome"]
            rec = {"prompt": build_prompt(row), "completion": build_completion(row["action"])}
            if split in ("train", "valid"):
                # SFT learns only from actions that worked; failures feed DPO.
                if out.get("is_error") or out.get("interrupted"):
                    counts["sft_excluded_failed"] += 1
                    continue
            else:
                rec["_tool"] = row["action"].get("tool")
                rec["_project"] = proj
            buckets[(proj, split)].append(rec)

    big = {p for p, n in repo_totals.items() if n >= args.min_repo_actions}

    # pooled = every repo, for the shared adapter
    for split in ("train", "valid"):
        rows = [r for (p, s), rs in buckets.items() if s == split for r in rs]
        counts[f"pooled_{split}"] = write_jsonl(os.path.join(DATA, "sft", "pooled", f"{split}.jsonl"), rows)

    for proj in sorted(big):
        for split in ("train", "valid"):
            rows = buckets.get((proj, split), [])
            counts[f"{proj}_{split}"] = write_jsonl(
                os.path.join(DATA, "sft", proj, f"{split}.jsonl"), rows)

    for proj in sorted(repo_totals):
        rows = buckets.get((proj, "test"), [])
        if rows:
            counts[f"test_{proj}"] = write_jsonl(os.path.join(DATA, "test", f"{proj}.jsonl"), rows)

    # DPO pairs, split on the same transcript boundary
    dpo = collections.defaultdict(list)
    if os.path.exists(args.pairs):
        with open(args.pairs, errors="ignore") as fh:
            for line in fh:
                p = json.loads(line)
                # Same transcript-level assignment as SFT, so a pair can never
                # come from a transcript held out for testing.
                split = split_of.get(p.get("src"), "train")
                if split == "test":
                    continue
                dpo[split].append({
                    "prompt": build_prompt(p),
                    "chosen": build_completion(p["chosen"]),
                    "rejected": build_completion(p["rejected"]),
                    "_tool": p["tool"], "_reason": p["reason"],
                    "_project": short_project(p["project"]),
                })
    for split, rows in dpo.items():
        counts[f"dpo_{split}"] = write_jsonl(os.path.join(DATA, "dpo", f"{split}.jsonl"), rows)

    print(f"transcripts       : {len(transcripts):,}")
    print(f"dropped ({'/'.join(DROP_TOOLS)}) : {dropped:,}")
    print(f"SFT rows excluded (failed/interrupted): {counts['sft_excluded_failed']:,}")
    print(f"\n{'repo':<16}{'actions':>9}{'train':>9}{'valid':>8}{'test':>8}  adapter")
    for proj in sorted(repo_totals, key=lambda p: -repo_totals[p]):
        tr = len(buckets.get((proj, 'train'), []))
        va = len(buckets.get((proj, 'valid'), []))
        te = len(buckets.get((proj, 'test'), []))
        tag = "own" if proj in big else "pooled-only"
        print(f"{proj:<16}{repo_totals[proj]:>9,}{tr:>9,}{va:>8,}{te:>8,}  {tag}")
    print(f"\npooled train/valid: {counts['pooled_train']:,} / {counts['pooled_valid']:,}")
    print(f"dpo train/valid   : {counts.get('dpo_train', 0):,} / {counts.get('dpo_valid', 0):,}")


if __name__ == "__main__":
    main()
