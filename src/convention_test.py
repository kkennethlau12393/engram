"""Does the state hold PER-REPO conventions, and condition on cwd?

This targets the skill half of the result, which is the larger half: not
recalling a specific past command, but having absorbed how a given repo works
and applying it to a situation never seen.

Repo conventions are read from repos.json (see repos.example.json), because they
are specific to whoever's transcripts are being used. In the corpus this was
developed on, three repos had unambiguous signatures: one TypeScript repo with
3846 `bun` calls and zero `pnpm`, one Python/Postgres repo with 18755 `python3`
and 4997 `psql` and zero `bun`, and one Rust repo using `cargo`.

Two tests:
  A  CONTROLLED  identical task text, only cwd differs. Out of distribution but
     a clean contrast: does the emitted command track the repo?
  B  REAL        held-out prompts from each repo whose gold command uses a known
     runner. Does the prediction use the same runner family? In distribution.
"""

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_eval import parse_action, cmd_of, parse_gold  # noqa: E402
from run_stream import load_stream  # noqa: E402
from state import StatefulLM  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

def _load_repos():
    """repos.json maps your project keys -> {cwd, expect:[runners]}. Gitignored;
    copy repos.example.json and fill in your own."""
    here = os.path.dirname(__file__)
    for cand in ("repos.json", "repos.example.json"):
        path = os.path.join(here, "..", cand)
        if os.path.exists(path):
            cfg = json.load(open(path))
            return ({k: v["cwd"] for k, v in cfg.items()},
                    {k: set(v["expect"]) for k, v in cfg.items()})
    return {}, {}


CWDS, EXPECT = _load_repos()

FAMILIES = {
    "bun": {"bun", "bunx"}, "pnpm": {"pnpm"}, "npm": {"npm"}, "npx": {"npx"},
    "python": {"python3", "python"}, "psql": {"psql"}, "cargo": {"cargo"},
    "yarn": {"yarn"}, "deno": {"deno"}, "git": {"git"},
}

TASKS = [
    "run the tests",
    "run the test suite and show failures",
    "install dependencies",
    "typecheck the project",
    "check what changed",
]


def runners_in(cmd):
    found = set()
    for fam, words in FAMILIES.items():
        for w in words:
            if re.search(rf"(?:^|[\s;&|(]){re.escape(w)}\b", cmd or ""):
                found.add(fam)
    return found


def synth_prompt(cwd, task):
    return f"cwd: {cwd}\nbranch: main\nrecent:\n[user] {task}\nNEXT_ACTION:"


def test_controlled(lm, repos, max_tokens):
    print(f"\n  {'=' * 74}\n  TEST A - CONTROLLED: identical task, only cwd differs")
    hits = collections.Counter()
    for task in TASKS:
        print(f"\n  task: {task!r}")
        for repo in repos:
            out = lm.read(synth_prompt(CWDS[repo], task), max_tokens=max_tokens)
            act = parse_action(out)
            cmd = cmd_of(act)
            fams = runners_in(cmd)
            want = EXPECT[repo]
            ok = bool(fams & {f for f in FAMILIES if FAMILIES[f] & want})
            hits[repo, ok] += 1
            print(f"    {'OK ' if ok else '   '}{repo:<14} {str(act['tool']):<9} {cmd[:74]!r}")
    print(f"\n  controlled hit-rate by repo:")
    for repo in repos:
        n = hits[repo, True] + hits[repo, False]
        if n:
            print(f"    {repo:<14} {hits[repo, True]}/{n}")
    return hits


def test_real(lm, repos, n_per, max_tokens):
    print(f"\n  {'=' * 74}\n  TEST B - REAL held-out prompts (in distribution)")
    summary = {}
    for repo in repos:
        stream = load_stream(os.path.join(DATA, "actions.jsonl"), repo)
        # Take from the END of the stream: most recent, least likely written in
        # the first 400 that trained the state.
        cands = []
        for it in reversed(stream):
            if it["tool"] != "Bash":
                continue
            g = parse_gold(it)
            gf = runners_in(cmd_of(g))
            if gf & set(FAMILIES):
                cands.append((it, gf))
            if len(cands) >= n_per:
                break
        match = exact_fam = 0
        for it, goldfams in cands:
            out = lm.read(it["prompt"], max_tokens=max_tokens)
            predfams = runners_in(cmd_of(parse_action(out)))
            if predfams & goldfams:
                match += 1
            want = {f for f in FAMILIES if FAMILIES[f] & EXPECT.get(repo, set())}
            if predfams & want:
                exact_fam += 1
        n = len(cands)
        summary[repo] = (match, exact_fam, n)
        print(f"    {repo:<14} runner matches gold {match}/{n}"
              f"   uses repo-typical runner {exact_fam}/{n}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    ap.add_argument("--state", default=None)
    ap.add_argument("--repos", default=",".join(sorted(CWDS)),
                    help="comma-separated keys from repos.json")
    ap.add_argument("--n-real", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    repos = [r for r in args.repos.split(",") if r in CWDS]
    lm = StatefulLM(model_id=args.model, max_len=args.max_len)
    tag = "BASE (no state)"
    if args.state:
        b = lm.state_norm()
        lm.load_state(args.state)
        if lm.state_norm() - b < 1e-6:
            raise SystemExit(f"load_state({args.state}) changed nothing; refusing to report.")
        tag = f"STATEFUL ({os.path.basename(args.state)})"
    print(f"\n{tag}  repos={repos}")

    test_controlled(lm, repos, args.max_tokens)
    test_real(lm, repos, args.n_real, args.max_tokens)


if __name__ == "__main__":
    main()
