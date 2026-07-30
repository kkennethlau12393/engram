"""Drive the experience stream through a stateful model and measure what sticks.

The loop is the whole thesis: write every interaction into the weights, never
into the prompt. Then ask two questions that decide whether this works at all.

  1. Does one write stick?      loss on an experience before vs right after writing it
  2. Does it survive?           loss on that same experience after N further writes

The control is what makes the numbers mean anything. HELD experiences are drawn
from the same stream but never written. If written and held items improve
equally, the model only learned the output format -- that is not memory. Memory
is the GAP between them.

  results/<run>.json   full curves
"""

import argparse
import collections
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from format import build_prompt, build_completion, short_project  # noqa: E402

# state/capability import mlx, which lives only in the venv. retrieve.py runs on
# SYSTEM python3 (for sentence-transformers) and imports load_stream/make_split
# from here, so these are imported lazily inside main().

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
RESULTS = os.path.join(HERE, "..", "results")


def load_stream(actions_path, project=None, limit=None):
    """Time-ordered experiences. Order matters: this is a stream, not a dataset."""
    rows = []
    with open(actions_path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["action"].get("tool") == "StructuredOutput":
                continue
            if project and short_project(r["project"]) != project:
                continue
            if r["outcome"].get("is_error") or r["outcome"].get("interrupted"):
                continue  # don't write known-bad actions into memory
            rows.append(r)
    rows.sort(key=lambda r: r.get("ts") or "")
    if limit:
        rows = rows[:limit]
    return [
        {"prompt": build_prompt(r), "completion": build_completion(r["action"]),
         "tool": r["action"]["tool"], "ts": r.get("ts"), "project": short_project(r["project"])}
        for r in rows
    ]


def probe(lm, items):
    return [lm.score(it["prompt"], it["completion"]) for it in items]


def make_split(stream, n, n_probe, n_held, seed):
    """Deterministic tracked/held split, shared with gen_eval so generation is
    scored on exactly the items the run wrote (and exactly the ones it didn't)."""
    rng = random.Random(seed)
    idx = list(range(len(stream)))
    held_idx = set(rng.sample(idx[: n + n_held], min(n_held, len(idx))))
    write_seq = [i for i in idx if i not in held_idx][:n]
    held = [stream[i] for i in sorted(held_idx)]
    step = max(1, len(write_seq) // n_probe)
    probe_pos = {write_seq[i]: i for i in range(0, len(write_seq), step)}
    tracked = [stream[i] for i in sorted(probe_pos)][:n_probe]
    written_at = [probe_pos[i] for i in sorted(probe_pos)][:n_probe]
    return write_seq, tracked, held, written_at, probe_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    ap.add_argument("--project", default=None, help="project key from extract.py --stats")
    ap.add_argument("--n", type=int, default=400, help="experiences to write")
    ap.add_argument("--n-probe", type=int, default=16, help="written items tracked")
    ap.add_argument("--n-held", type=int, default=16, help="never-written controls")
    ap.add_argument("--every", type=int, default=50, help="writes between probe rounds")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=1, help="gradient steps per write")
    ap.add_argument("--optimizer", default="adamw")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--name", default=None)
    ap.add_argument("--interfere-project", default=None,
                    help="after the main stream, keep writing from THIS project while "
                         "still probing the first project's memories -- the real "
                         "catastrophic-forgetting test (unrelated task interference)")
    ap.add_argument("--n-interfere", type=int, default=400)
    ap.add_argument("--replay-frac", type=float, default=0.0,
                    help="fraction of interference steps that instead replay an OLD "
                         "experience from the original project. The standard continual-"
                         "learning defence against catastrophic interference.")
    ap.add_argument("--require-runner", default=None,
                    help="keep only Bash experiences whose command uses one of these "
                         "comma-separated words. The natural stream is too sparse to test "
                         "convention learning: a given runner appeared in only 6/400 consecutive "
                         "experiences, so a null result there measures nothing.")
    ap.add_argument("--save-state", action="store_true",
                    help="persist the final LoRA state so it can be re-probed or resumed")
    args = ap.parse_args()

    from state import StatefulLM
    import capability

    name = args.name or f"{args.project}-n{args.n}-lr{args.lr}-r{args.rank}-s{args.steps}"
    os.makedirs(RESULTS, exist_ok=True)

    stream = load_stream(os.path.join(DATA, "actions.jsonl"), args.project)
    if args.require_runner:
        import re as _re
        words = [w.strip() for w in args.require_runner.split(",") if w.strip()]
        pat = _re.compile(r"(?:^|[\s;&|(])(" + "|".join(_re.escape(w) for w in words) + r")\b")
        before = len(stream)
        stream = [s_ for s_ in stream if s_["tool"] == "Bash" and pat.search(s_["completion"])]
        print(f"  runner filter {words}: {before:,} -> {len(stream):,} experiences")
    if len(stream) < args.n + args.n_held + 10:
        print(f"only {len(stream)} experiences for project={args.project}", file=sys.stderr)
    # HELD items are pulled out first so they can never be written.
    write_seq, tracked, held, tracked_written_at, probe_pos = make_split(
        stream, args.n, args.n_probe, args.n_held, args.seed)

    print(f"run={name}")
    print(f"  model={args.model} rank={args.rank} layers={args.num_layers} "
          f"lr={args.lr} steps/write={args.steps} opt={args.optimizer}")
    print(f"  stream={len(stream):,} writes={len(write_seq)} tracked={len(tracked)} held={len(held)}")
    print(f"  tools in stream: {dict(collections.Counter(s['tool'] for s in stream).most_common(6))}")

    lm = StatefulLM(model_id=args.model, rank=args.rank, num_layers=args.num_layers,
                    lr=args.lr, optimizer=args.optimizer, max_len=args.max_len)

    t_start = time.time()
    base_tracked = probe(lm, tracked)
    base_held = probe(lm, held)
    cap_before = capability.measure(lm)
    print(f"\n  baseline loss  tracked={statistics.mean(base_tracked):.3f}  "
          f"held={statistics.mean(base_held):.3f}  "
          f"capability={statistics.mean(cap_before):.3f}   ({time.time() - t_start:.0f}s)")

    curve = []
    immediate = []   # (loss before write, loss right after write)
    t0 = time.time()

    for n, i in enumerate(write_seq, 1):
        item = stream[i]
        if i in probe_pos:
            before = lm.score(item["prompt"], item["completion"])
            lm.write(item["prompt"], item["completion"], steps=args.steps)
            after = lm.score(item["prompt"], item["completion"])
            immediate.append({"n": n, "before": before, "after": after})
        else:
            lm.write(item["prompt"], item["completion"], steps=args.steps)

        if n % args.every == 0 or n == len(write_seq):
            tr = probe(lm, tracked)
            hd = probe(lm, held)
            row = {
                "writes": n,
                "tracked_mean": statistics.mean(tr),
                "held_mean": statistics.mean(hd),
                "tracked": tr,
                "held": hd,
                "state_norm": lm.state_norm(),
                "elapsed_s": time.time() - t0,
            }
            curve.append(row)
            gap = row["held_mean"] - row["tracked_mean"]
            print(f"  writes={n:5d}  tracked={row['tracked_mean']:.3f}  "
                  f"held={row['held_mean']:.3f}  gap={gap:+.3f}  "
                  f"|state|={row['state_norm']:.2f}  {row['elapsed_s'] / n:.2f}s/write")

    # Interference phase: keep writing, but from a DIFFERENT repo. The tracked
    # items are never revisited, so any advantage they keep is memory surviving
    # unrelated writes -- which is what catastrophic forgetting actually means.
    interfere = []
    if args.interfere_project:
        other = load_stream(os.path.join(DATA, "actions.jsonl"), args.interfere_project)
        other = other[: args.n_interfere]
        print(f"\n  INTERFERENCE: {len(other)} writes from '{args.interfere_project}' "
              f"while probing '{args.project}' memories")
        base_gap = statistics.mean(base_held) - statistics.mean(base_tracked)

        # Replay pool EXCLUDES the tracked probes. Replaying the very items being
        # measured would make "memory survived" mean nothing -- it would just be
        # rewriting the answers. Survival must come from replaying OTHER
        # experiences holding the region of weight space those memories live in.
        tracked_ids = {id(stream[i]) for i in probe_pos}
        replay_pool = [stream[i] for i in write_seq if id(stream[i]) not in tracked_ids]
        rp_rng = random.Random(args.seed + 1)
        n_replayed = 0

        for n, item in enumerate(other, 1):
            if args.replay_frac and replay_pool and rp_rng.random() < args.replay_frac:
                old_item = rp_rng.choice(replay_pool)
                lm.write(old_item["prompt"], old_item["completion"], steps=args.steps)
                n_replayed += 1
            else:
                lm.write(item["prompt"], item["completion"], steps=args.steps)
            if n % args.every == 0 or n == len(other):
                tr, hd = probe(lm, tracked), probe(lm, held)
                row = {"interfere_writes": n,
                       "tracked_mean": statistics.mean(tr), "held_mean": statistics.mean(hd),
                       "tracked": tr, "held": hd, "state_norm": lm.state_norm()}
                interfere.append(row)
                gap = (row["held_mean"] - row["tracked_mean"]) - base_gap
                row["n_replayed"] = n_replayed
                print(f"    +{n:5d} foreign  tracked={row['tracked_mean']:.3f}  "
                      f"held={row['held_mean']:.3f}  gap={gap:+.3f}  replays={n_replayed}")

    cap_after = capability.measure(lm)

    out = {
        "name": name,
        "capability": {"before": cap_before, "after": cap_after},
        "interfere": interfere,
        "replay_frac": args.replay_frac,
        "interfere_project": args.interfere_project,
        "config": vars(args),
        "baseline": {"tracked": base_tracked, "held": base_held,
                     "tracked_mean": statistics.mean(base_tracked),
                     "held_mean": statistics.mean(base_held)},
        "tracked_written_at": tracked_written_at,
        "curve": curve,
        "immediate": immediate,
        "total_s": time.time() - t_start,
    }
    if args.save_state:
        sp = os.path.join(HERE, "..", "adapters", f"{name}.safetensors")
        lm.save(sp)
        print(f"\n  saved state -> {sp}  ({lm.n_writes} writes)")

    dest = os.path.join(RESULTS, f"{name}.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)

    # Summary: the three numbers that decide whether this works.
    fin = curve[-1]
    d_tracked = statistics.mean(base_tracked) - fin["tracked_mean"]
    d_held = statistics.mean(base_held) - fin["held_mean"]
    print(f"\n  {'=' * 62}")
    print(f"  written items improved by : {d_tracked:+.3f} nats/token")
    print(f"  held  items improved by   : {d_held:+.3f} nats/token   <- format learning")
    print(f"  MEMORY (gap)              : {d_tracked - d_held:+.3f}   <- the actual claim")
    if immediate:
        di = statistics.mean(x["before"] - x["after"] for x in immediate)
        print(f"  one-write drop (immediate): {di:+.3f}")
    capability.report(cap_before, cap_after)
    print(f"\n  wrote {dest}")


if __name__ == "__main__":
    main()
