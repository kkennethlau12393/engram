"""Pool independent seeds into one effect estimate for the memory claim.

Needed because the first runs were badly underpowered: 16 probe items against a
per-item sd of ~0.9 gives SE 0.28, a confidence interval over a nat wide. Worse,
two of those runs shared a seed, so they scored the SAME 16 items -- their
agreement was determinism, not replication.

Here each seed draws a different tracked/held split, so pooling across seeds is
genuine replication. Reports the per-seed effect, the pooled effect with a
confidence interval, and a paired t-test against zero.
"""

import argparse
import glob
import json
import math
import os
import statistics as st

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def effect(path):
    r = json.load(open(path))
    b, fin = r["baseline"], r["curve"][-1]
    dt = [x - y for x, y in zip(b["tracked"], fin["tracked"])]
    dh = [x - y for x, y in zip(b["held"], fin["held"])]
    se = math.sqrt(st.variance(dt) / len(dt) + st.variance(dh) / len(dh))
    cap = r.get("capability")
    return {
        "name": r["name"],
        "seed": r["config"]["seed"],
        "n_t": len(dt), "n_h": len(dh),
        "mean_t": st.mean(dt), "mean_h": st.mean(dh),
        "sd_t": st.stdev(dt), "sd_h": st.stdev(dh),
        "effect": st.mean(dt) - st.mean(dh),
        "se": se,
        "cap_delta": (st.mean(cap["after"]) - st.mean(cap["before"])) if cap else None,
        "drift": fin["state_norm"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="pow-s*.json")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(RESULTS, args.glob)))
    if not paths:
        print(f"no results matching {args.glob}")
        return
    rows = [effect(p) for p in paths]

    # Guards. Pooling is only meaningful across INDEPENDENT draws of the split
    # from the SAME configuration, and only over runs that did not diverge.
    problems = []
    seeds = [e["seed"] for e in rows]
    if len(set(seeds)) != len(seeds):
        problems.append(
            f"duplicate seeds {sorted(seeds)}: these score the same items, so they are "
            "re-executions, not replications -- pooling them fakes precision")
    cfgs = {(json.load(open(p))["config"]["lr"],
             json.load(open(p))["config"]["steps"],
             json.load(open(p))["config"]["n"]) for p in paths}
    if len(cfgs) > 1:
        problems.append(f"mixed configurations {sorted(cfgs)}: not the same experiment")
    diverged = [e["name"] for e in rows if e["mean_h"] < 0]
    if diverged:
        problems.append(
            f"diverged run(s) {diverged}: held loss got WORSE, so the 'effect' is a "
            "gap between two broken numbers")
    if problems:
        print("\n  REFUSING TO POOL:")
        for p in problems:
            print(f"    - {p}")
        print("\n  per-run figures only:")

    print(f"\n  {'run':<12}{'seed':>5}{'n':>6}{'written':>10}{'held':>9}"
          f"{'effect':>9}{'SE':>7}{'cap':>8}")
    for e in rows:
        cap = f"{e['cap_delta']:+.3f}" if e["cap_delta"] is not None else "   -"
        print(f"  {e['name']:<12}{e['seed']:>5}{e['n_t']:>6}{e['mean_t']:>+10.3f}"
              f"{e['mean_h']:>+9.3f}{e['effect']:>+9.3f}{e['se']:>7.3f}{cap:>8}")

    effs = [e["effect"] for e in rows]
    pooled = st.mean(effs)
    if len(effs) > 1:
        # Between-seed SE: each seed is an independent draw of the split.
        se_between = st.stdev(effs) / math.sqrt(len(effs))
        # Also pool the within-seed SEs, and take the larger (conservative).
        se_within = math.sqrt(sum(e["se"] ** 2 for e in rows)) / len(rows)
        se = max(se_between, se_within)
        t = pooled / se if se else float("inf")
        lo, hi = pooled - 1.96 * se, pooled + 1.96 * se
        print(f"\n  POOLED EFFECT  {pooled:+.3f}   SE {se:.3f}"
              f"   95% CI [{lo:+.3f}, {hi:+.3f}]   t={t:.2f}")
        print(f"    (between-seed SE {se_between:.3f}, within-seed SE {se_within:.3f})")
        verdict = ("MEMORY IS REAL" if lo > 0 else
                   "NOT RESOLVED - CI includes zero" if hi > 0 else
                   "EFFECT IS NEGATIVE")
        print(f"    VERDICT: {verdict}")
        total_n = sum(e["n_t"] for e in rows)
        print(f"    based on {total_n} written / {sum(e['n_h'] for e in rows)} held items "
              f"across {len(rows)} independent splits")
    caps = [e["cap_delta"] for e in rows if e["cap_delta"] is not None]
    if caps:
        print(f"\n  CAPABILITY delta (positive = degraded): {st.mean(caps):+.3f}")


if __name__ == "__main__":
    main()
