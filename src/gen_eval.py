"""Does the stateful model actually EMIT the right action?

Every recall number so far is teacher-forced loss. Loss dropping on
`pnpm vitest run` is not the same as the model producing `pnpm vitest run`
instead of `npm test`. This generates and compares against what Claude really
did next.

Scored on the SAME split the run used:
  tracked = experiences written into state   -> recall
  held    = never written                    -> control for general improvement

A gain on tracked that does not appear on held is memory changing behaviour.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from run_stream import load_stream, make_split  # noqa: E402
from state import StatefulLM  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
RESULTS = os.path.join(HERE, "..", "results")

TOOL_RE = re.compile(r'"tool"\s*:\s*"([A-Za-z_]+)"')


def parse_action(text):
    """Model output -> {tool, input}. Falls back to a regex for the tool name so a
    truncated generation still scores on the primary metric."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text).replace("```", "")
    start = text.find("{")
    if start >= 0:
        # Walk back from the end for the longest parseable prefix-object.
        for end in range(len(text), start, -1):
            try:
                obj = json.loads(text[start:end])
                if isinstance(obj, dict) and "tool" in obj:
                    return {"tool": obj.get("tool"),
                            "input": obj.get("input") if isinstance(obj.get("input"), dict) else {}}
            except Exception:
                continue
    m = TOOL_RE.search(text)
    return {"tool": m.group(1), "input": {}} if m else {"tool": None, "input": {}}


def parse_gold(item):
    """Gold completions are truncated to a char budget by format.build_completion,
    so long actions (big Write payloads) are not valid JSON. The tool name is
    carried alongside on the stream record; recover the command with a regex."""
    try:
        obj = json.loads(item["completion"])
        return {"tool": obj.get("tool"),
                "input": obj.get("input") if isinstance(obj.get("input"), dict) else {}}
    except Exception:
        pass
    txt = item["completion"]
    tool = item.get("tool") or (TOOL_RE.search(txt).group(1) if TOOL_RE.search(txt) else None)
    for key in ("command", "file_path", "url", "pattern"):
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)', txt)
        if m:
            try:
                return {"tool": tool, "input": {key: json.loads(f'"{m.group(1)}"')}}
            except Exception:
                return {"tool": tool, "input": {key: m.group(1)}}
    return {"tool": tool, "input": {}}


def cmd_of(action):
    inp = action.get("input") or {}
    if not isinstance(inp, dict):
        return ""
    for k in ("command", "file_path", "url", "pattern"):
        if isinstance(inp.get(k), str):
            return inp[k]
    return ""


def toks(s):
    return re.findall(r"[A-Za-z0-9_./@-]+", (s or "").lower())


def f1(a, b):
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return 1.0 if not ta and not tb else 0.0
    inter = 0
    pool = list(tb)
    for t in ta:
        if t in pool:
            pool.remove(t)
            inter += 1
    if inter == 0:
        return 0.0
    p, r = inter / len(ta), inter / len(tb)
    return 2 * p * r / (p + r)


RAG_HEADER = "Similar past actions in this repository:\n"


def rag_prompt(item, k):
    """Same memories, delivered through context instead of weights."""
    parts = [RAG_HEADER]
    for j, nb in enumerate(item["neighbors"][:k], 1):
        parts.append(f"\n--- past action {j}\n{nb['prompt']}\n=> {nb['completion']}\n")
    parts.append("\nNow predict the next action for the current state.\n\n")
    parts.append(item["prompt"])
    return "".join(parts)


def evaluate(lm, items, label, max_tokens, show=0, rag_k=0):
    n = len(items)
    ok_json = tool_hit = exact = 0
    f1s = []
    samples = []
    t0 = time.time()
    for i, it in enumerate(items):
        text = rag_prompt(it, rag_k) if rag_k else it["prompt"]
        out = lm.read(text, max_tokens=max_tokens)
        pred = parse_action(out)
        gold = parse_gold(it)
        if pred["tool"]:
            ok_json += 1
        if pred["tool"] == gold.get("tool"):
            tool_hit += 1
        gc, pc = cmd_of(gold), cmd_of(pred)
        if pred["tool"] == gold.get("tool") and gc and pc and gc.strip() == pc.strip():
            exact += 1
        if gold.get("tool") == "Bash":
            f1s.append(f1(pc, gc) if pred["tool"] == "Bash" else 0.0)
        if len(samples) < show:
            samples.append((gold.get("tool"), gc, pred["tool"], pc))
    dt = time.time() - t0
    res = {
        "label": label, "n": n,
        "parseable": ok_json / n,
        "tool_acc": tool_hit / n,
        "exact": exact / n,
        "bash_f1": statistics.mean(f1s) if f1s else None,
        "n_bash": len(f1s),
        "secs": dt,
    }
    bf = f"{res['bash_f1']:.3f}" if res["bash_f1"] is not None else "  n/a"
    print(f"  {label:<28} parse {res['parseable']:.2f}  tool {res['tool_acc']:.3f}  "
          f"exact {res['exact']:.3f}  bashF1 {bf} (n={res['n_bash']})  {dt / n:.1f}s/item")
    return res, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    ap.add_argument("--project", default=None, help="project key from extract.py --stats")
    ap.add_argument("--state", default=None, help="saved LoRA state; omit for the base model")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--n-held", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--name", default="gen")
    ap.add_argument("--rag", default=None,
                    help="data/rag/<name>.jsonl from retrieve.py. Injects the SAME "
                         "memories into context and uses the BASE model -- the "
                         "stateless-context comparison the premise rests on.")
    ap.add_argument("--rag-k", type=int, default=5)
    ap.add_argument("--max-len-rag", type=int, default=2048,
                    help="RAG prompts are far longer; give them room or the "
                         "comparison is rigged by truncation")
    args = ap.parse_args()

    if args.rag:
        rows = [json.loads(l) for l in open(args.rag) if l.strip()]
        tracked = [r for r in rows if r["split"] == "tracked"]
        held = [r for r in rows if r["split"] == "held"]
    else:
        stream = load_stream(os.path.join(DATA, "actions.jsonl"), args.project)
        _, tracked, held, _, _ = make_split(
            stream, args.n, args.n_probe, args.n_held, args.seed)

    lm = StatefulLM(model_id=args.model, rank=args.rank, num_layers=args.num_layers,
                    max_len=args.max_len_rag if args.rag else args.max_len)
    tag = "BASE (no state)"
    if args.state:
        # Verify the load actually moved the weights. A silent no-op would make
        # the stateful run identical to base and read as "memory does nothing".
        before = lm.state_norm()
        lm.load_state(args.state)
        after = lm.state_norm()
        if after - before < 1e-6:
            raise SystemExit(
                f"load_state({args.state}) did not change any weights "
                f"(drift {before:.4f} -> {after:.4f}). Refusing to report a bogus null result."
            )
        tag = f"STATEFUL ({os.path.basename(args.state)}, drift {after:.2f})"

    print(f"\n{tag}   project={args.project}  tracked={len(tracked)} held={len(held)}")
    k = args.rag_k if args.rag else 0
    if args.rag:
        tag += f" + RAG k={k} (base weights, memories in context)"
        print(f"  {tag}")
    r_tr, samples = evaluate(lm, tracked, "tracked (written)", args.max_tokens,
                             show=args.show, rag_k=k)
    r_hd, _ = evaluate(lm, held, "held (never written)", args.max_tokens, rag_k=k)

    print(f"\n  tool-acc gap (tracked - held): {r_tr['tool_acc'] - r_hd['tool_acc']:+.3f}")
    if r_tr["bash_f1"] is not None and r_hd["bash_f1"] is not None:
        print(f"  bashF1   gap (tracked - held): {r_tr['bash_f1'] - r_hd['bash_f1']:+.3f}")

    if samples:
        print(f"\n  {'-' * 70}\n  SAMPLES (gold vs predicted)")
        for gt, gc, pt, pc in samples:
            mark = "OK " if gt == pt else "XX "
            print(f"  {mark}gold {gt:<10} {gc[:60]!r}")
            print(f"      pred {str(pt):<10} {pc[:60]!r}")

    dest = os.path.join(RESULTS, f"{args.name}.json")
    with open(dest, "w") as fh:
        json.dump({"state": args.state, "tracked": r_tr, "held": r_hd,
                   "config": vars(args)}, fh, indent=2)
    print(f"\n  wrote {dest}")


if __name__ == "__main__":
    main()
