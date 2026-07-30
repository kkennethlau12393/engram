"""Extract next-action training records from Claude Code transcripts.

Transcripts are trees, not lists: rewinds and branches mean a session file holds
several divergent histories linked by parentUuid. For each tool_use we walk its
own ancestor chain, so every action is paired with the context that actually
preceded it on its branch.

Emits one JSON record per action to data/actions.jsonl.
"""

import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.expanduser("~/.claude/projects")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "actions.jsonl")

# Observations are the reason the corpus is 2.2 GB; a single Bash result can run
# to hundreds of KB. Cap hard, keeping head and tail (errors usually surface at
# the tail, the command echo at the head).
OBS_HEAD = 1200
OBS_TAIL = 600
CTX_TURNS = 12  # ancestor records kept as context
INTERRUPT_MARK = "[Request interrupted by user"


def tier_of(path: str) -> str:
    """main = <proj>/<uuid>.jsonl, sub = nested under a session's subagents/."""
    if "/subagents/" in path:
        return "workflow" if "/workflows/" in path else "subagent"
    return "main"


def discover(root: str = ROOT):
    for p in glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True):
        if os.path.basename(p) == "journal.jsonl":
            continue  # workflow bookkeeping, not a trajectory
        rel = os.path.relpath(p, root)
        yield p, rel.split(os.sep)[0], tier_of(p)


def load_records(path: str):
    out = []
    try:
        with open(path, errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


def content_of(rec):
    msg = rec.get("message")
    if isinstance(msg, dict):
        return msg.get("content")
    return None


def clip(text: str) -> str:
    text = text if isinstance(text, str) else json.dumps(text, default=str)
    if len(text) <= OBS_HEAD + OBS_TAIL:
        return text
    return f"{text[:OBS_HEAD]}\n…[{len(text) - OBS_HEAD - OBS_TAIL} chars elided]…\n{text[-OBS_TAIL:]}"


def result_text(block, rec):
    """Prefer the structured toolUseResult; fall back to the content block."""
    tur = rec.get("toolUseResult")
    if isinstance(tur, dict):
        for key in ("stdout", "output", "content", "stderr", "text"):
            val = tur.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return json.dumps(tur, default=str)[:4000]
    body = block.get("content")
    if isinstance(body, list):
        return "\n".join(
            b.get("text", "") for b in body if isinstance(b, dict) and b.get("type") == "text"
        )
    return body if isinstance(body, str) else json.dumps(body, default=str)


def has_tool_use(rec):
    body = content_of(rec)
    return rec.get("type") == "assistant" and isinstance(body, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_use" for b in body
    )


def index_file(recs):
    """Build uuid -> record, tool_use_id -> outcome, and the set of interrupted uuids."""
    by_uuid, outcomes, interrupt_recs = {}, {}, []
    for rec in recs:
        uid = rec.get("uuid")
        if uid:
            by_uuid[uid] = rec
        body = content_of(rec)
        if rec.get("type") == "user" and isinstance(body, list):
            for block in body:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    outcomes[block.get("tool_use_id")] = {
                        "is_error": bool(block.get("is_error")),
                        "text": clip(result_text(block, rec)),
                    }
            if INTERRUPT_MARK in str(body):
                interrupt_recs.append(rec)

    # An interrupt record's parentUuid points at an attachment or another user
    # record, not at the action that got cut off. Walk up to the nearest
    # assistant turn that actually issued a tool_use and blame that one.
    interrupted = set()
    for rec in interrupt_recs:
        cur, hops = rec.get("parentUuid"), 0
        while cur and cur in by_uuid and hops < 8:
            anc = by_uuid[cur]
            if has_tool_use(anc):
                interrupted.add(anc["uuid"])
                break
            cur, hops = anc.get("parentUuid"), hops + 1
    return by_uuid, outcomes, interrupted


def render_ctx(rec, outcomes):
    """One context line per ancestor record; None when it carries no signal."""
    body = content_of(rec)
    kind = rec.get("type")
    if kind == "user":
        if isinstance(body, str):
            txt = body.strip()
            if not txt or txt.startswith("<"):
                return None  # system-reminder envelopes, not human speech
            return f"[user] {clip(txt)}"
        if isinstance(body, list):
            lines = []
            for block in body:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    out = outcomes.get(block.get("tool_use_id"), {})
                    tag = "error" if out.get("is_error") else "ok"
                    lines.append(f"[result:{tag}] {out.get('text', '')}")
                elif block.get("type") == "text" and block.get("text", "").strip():
                    lines.append(f"[user] {clip(block['text'])}")
            return "\n".join(lines) or None
    if kind == "assistant" and isinstance(body, list):
        lines = []
        for block in body:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text", "").strip():
                lines.append(f"[assistant] {clip(block['text'])}")
            elif block.get("type") == "tool_use":
                lines.append(
                    f"[action] {block.get('name')} {json.dumps(block.get('input', {}), default=str)[:800]}"
                )
        return "\n".join(lines) or None
    return None


def ancestors(rec, by_uuid, limit=CTX_TURNS):
    """Walk parentUuid up this branch; returns oldest-first."""
    chain, seen, cur = [], set(), rec.get("parentUuid")
    while cur and cur in by_uuid and cur not in seen and len(chain) < limit * 2:
        seen.add(cur)
        chain.append(by_uuid[cur])
        cur = by_uuid[cur].get("parentUuid")
    return list(reversed(chain))


def extract_file(path, project, tier):  # noqa: C901 - path is used in emitted records
    recs = load_records(path)
    if not recs:
        return
    by_uuid, outcomes, interrupted = index_file(recs)

    for rec in recs:
        if rec.get("type") != "assistant":
            continue
        body = content_of(rec)
        if not isinstance(body, list):
            continue

        # Context is shared by every tool_use in this assistant turn; build once.
        ctx = None
        for block in body:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if ctx is None:
                lines = []
                for anc in ancestors(rec, by_uuid):
                    rendered = render_ctx(anc, outcomes)
                    if rendered:
                        lines.append(rendered)
                ctx = lines[-CTX_TURNS:]

            out = outcomes.get(block.get("id"), {})
            yield {
                "project": project,
                "tier": tier,
                # Subagent transcripts inherit the parent sessionId, so sessionId
                # does NOT identify a trajectory. The file does.
                "src": os.path.relpath(path, ROOT),
                "session": rec.get("sessionId"),
                "uuid": rec.get("uuid"),
                "ts": rec.get("timestamp"),
                "cwd": rec.get("cwd"),
                "gitBranch": rec.get("gitBranch"),
                "context": ctx,
                "action": {"tool": block.get("name"), "input": block.get("input", {})},
                "outcome": {
                    "is_error": bool(out.get("is_error")),
                    "interrupted": rec.get("uuid") in interrupted,
                    "observed": bool(out),
                },
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--stats", action="store_true", help="print summary only, write nothing")
    args = ap.parse_args()

    per_project = collections.Counter()
    per_tier = collections.Counter()
    tools = collections.Counter()
    errors = collections.Counter()
    interrupts = 0
    total = 0
    no_ctx = 0

    files = list(discover())
    print(f"discovered {len(files)} transcripts", file=sys.stderr)

    sink = None if args.stats else open(args.out, "w")
    try:
        for i, (path, project, tier) in enumerate(files, 1):
            if i % 500 == 0:
                print(f"  {i}/{len(files)} files, {total:,} actions", file=sys.stderr)
            for row in extract_file(path, project, tier):
                total += 1
                per_project[project] += 1
                per_tier[tier] += 1
                tools[row["action"]["tool"]] += 1
                if row["outcome"]["is_error"]:
                    errors[project] += 1
                if row["outcome"]["interrupted"]:
                    interrupts += 1
                if not row["context"]:
                    no_ctx += 1
                if sink:
                    sink.write(json.dumps(row, default=str) + "\n")
    finally:
        if sink:
            sink.close()

    print(f"\nTOTAL ACTIONS: {total:,}")
    print(f"tool errors  : {sum(errors.values()):,}")
    print(f"interrupted  : {interrupts:,}")
    print(f"empty context: {no_ctx:,} ({no_ctx / max(total, 1) * 100:.1f}%)")
    print(f"\nby tier: {dict(per_tier)}")
    print(f"\n{'project':<45}{'actions':>9}{'errors':>8}")
    for proj, n in per_project.most_common():
        print(f"{proj:<45}{n:>9,}{errors[proj]:>8}")
    print(f"\ntop tools: {dict(tools.most_common(15))}")
    if sink is not None or not args.stats:
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
