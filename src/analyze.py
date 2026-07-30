"""Read a run's curves and answer the two questions that decide the design.

Everything is measured as an ADVANTAGE over the held-out control, never as raw
loss. Held items improve too -- the model gets better at emitting tool-call JSON
regardless of what it was written with. Raw loss therefore cannot distinguish
"remembered this experience" from "learned the output format". The advantage can:

    advantage_j(n) = [held_mean(n) - loss_j(n)] - [held_mean(0) - loss_j(0)]

i.e. how much item j pulled ahead of the control, over and above where it
started. Positive = item-specific memory. Decaying toward zero = forgetting.
"""

import argparse
import collections
import glob
import json
import os
import statistics

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def analyze(path):
    run = json.load(open(path))
    base = run["baseline"]
    curve = run["curve"]
    written_at = run["tracked_written_at"]
    n_tracked = min(len(written_at), len(base["tracked"]), len(curve[-1]["tracked"]))

    print(f"\n{'=' * 72}\nRUN: {run['name']}")
    cfg = run["config"]
    print(f"  {cfg['model']}  rank={cfg['rank']} layers={cfg['num_layers']} "
          f"lr={cfg['lr']} steps/write={cfg['steps']}")
    print(f"  {cfg['n']} writes on '{cfg['project']}'   "
          f"{run['total_s'] / 60:.0f} min total, {run['total_s'] / cfg['n']:.2f}s/write")

    # ---- headline: memory vs format learning ----
    fin = curve[-1]
    d_tracked = base["tracked_mean"] - fin["tracked_mean"]
    d_held = base["held_mean"] - fin["held_mean"]
    print(f"\n  LEARNING (nats/token improvement, start -> end)")
    print(f"    written items : {d_tracked:+.3f}")
    print(f"    held control  : {d_held:+.3f}   <- format learning, not memory")
    print(f"    MEMORY        : {d_tracked - d_held:+.3f}")

    if run.get("immediate"):
        imm = run["immediate"][:n_tracked]
        drops = [x["before"] - x["after"] for x in imm]
        print(f"\n  ONE-SHOT ACQUISITION (single gradient step, per item)")
        print(f"    mean drop {statistics.mean(drops):+.3f}   "
              f"median {statistics.median(drops):+.3f}   "
              f"min {min(drops):+.3f}   max {max(drops):+.3f}")
        print(f"    items where one write helped: "
              f"{sum(1 for d in drops if d > 0)}/{len(drops)}")

    # ---- forgetting: advantage vs age ----
    base_adv = [base["held_mean"] - base["tracked"][j] for j in range(n_tracked)]
    by_age = collections.defaultdict(list)
    for row in curve:
        n = row["writes"]
        for j in range(n_tracked):
            w = written_at[j]
            if n < w:
                continue  # not written yet
            adv = (row["held_mean"] - row["tracked"][j]) - base_adv[j]
            by_age[n - w].append(adv)

    if by_age:
        print(f"\n  FORGETTING CURVE (advantage over control, by age)")
        print(f"    {'writes since written':<24}{'advantage':>11}{'n':>6}  {'from items written at':<22}")
        buckets = [(0, 0), (1, 50), (51, 100), (101, 200), (201, 300), (301, 10**9)]
        # Track WHICH items land in each bucket: only early-written items can
        # reach old ages, so a raw bucket mean compares different populations.
        contrib = collections.defaultdict(list)
        for row in curve:
            for j in range(n_tracked):
                w = written_at[j]
                if row["writes"] < w:
                    continue
                age = row["writes"] - w
                for lo, hi in buckets:
                    if lo <= age <= hi:
                        contrib[(lo, hi)].append(w)
        for lo, hi in buckets:
            vals = [v for a, vs in by_age.items() if lo <= a <= hi for v in vs]
            if not vals:
                continue
            ws = contrib[(lo, hi)]
            label = "just written" if hi == 0 else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")
            span = f"writes {min(ws)}-{max(ws)}" if ws else "-"
            print(f"    {label:<24}{statistics.mean(vals):>+11.3f}{len(vals):>6}  {span:<22}")

        # Per-item retention: each item normalised to ITS OWN first-probe
        # advantage, so per-item difficulty and cohort effects divide out.
        print(f"\n  RETENTION, per-item normalised (1.00 = as good as when first probed)")
        ret_by_age = collections.defaultdict(list)
        for j in range(n_tracked):
            w = written_at[j]
            first = None
            for row in curve:
                if row["writes"] < w:
                    continue
                adv = (row["held_mean"] - row["tracked"][j]) - base_adv[j]
                if first is None:
                    first = adv
                    continue
                if abs(first) < 1e-6:
                    continue
                ret_by_age[row["writes"] - w].append(adv / first)
        for lo, hi in buckets[1:]:
            vals = [v for a, vs in ret_by_age.items() if lo <= a <= hi for v in vs]
            if vals:
                label = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
                print(f"    age {label:<20}{statistics.mean(vals):>10.2f}x{len(vals):>6}")

    # ---- did the gap widen or close over the run? ----
    print(f"\n  GAP TRAJECTORY (held_mean - tracked_mean, baseline-corrected)")
    b_gap = base["held_mean"] - base["tracked_mean"]
    for row in curve:
        g = (row["held_mean"] - row["tracked_mean"]) - b_gap
        bar = "#" * max(0, int(g * 40))
        print(f"    writes={row['writes']:<5} gap={g:+.3f}  |drift|={row['state_norm']:.3f}  {bar}")

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="result json files (default: all)")
    a = ap.parse_args()
    paths = a.runs or sorted(glob.glob(os.path.join(RESULTS, "*.json")))
    if not paths:
        print("no results yet")
        return
    for p in paths:
        try:
            analyze(p)
        except Exception as e:
            print(f"  [skip {os.path.basename(p)}: {e}]")


if __name__ == "__main__":
    main()
