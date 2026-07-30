"""Build the retrieval baseline: the same memories, consumed through context.

This is the comparison the whole premise rests on. The user's framing is that
normal LLMs use stateless context memory and this project replaces it with state
in the weights. That claim is untested until parametric and retrieval are given
the SAME memories and scored on the SAME items.

Fairness rule: the retrieval pool is exactly the 400 experiences the LoRA was
written with -- not the full 17K stream. Otherwise the comparison is corpus size
versus mechanism, which proves nothing.

Runs on SYSTEM python3 (sentence-transformers + torch live there; mlx lives in
the venv). Writes data/rag/<name>.jsonl consumed by gen_eval --rag.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_stream import load_stream, make_split  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_TAIL = 1000  # MiniLM truncates at 256 tokens; the prompt tail predicts best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--n-held", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--require-runner", default=None)
    ap.add_argument("--name", default="rag")
    args = ap.parse_args()

    stream = load_stream(os.path.join(DATA, "actions.jsonl"), args.project)
    if args.require_runner:
        import re
        words = [w.strip() for w in args.require_runner.split(",") if w.strip()]
        pat = re.compile(r"(?:^|[\s;&|(])(" + "|".join(re.escape(w) for w in words) + r")\b")
        stream = [s for s in stream if s["tool"] == "Bash" and pat.search(s["completion"])]

    write_seq, tracked, held, _, _ = make_split(
        stream, args.n, args.n_probe, args.n_held, args.seed)
    pool = [stream[i] for i in write_seq]   # exactly what the LoRA was written with

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, device="mps")

    def emb(rows):
        return model.encode([r["prompt"][-EMBED_TAIL:] for r in rows], batch_size=128,
                            convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=False)

    print(f"pool={len(pool)}  tracked={len(tracked)}  held={len(held)}", flush=True)
    e_pool = emb(pool)

    os.makedirs(os.path.join(DATA, "rag"), exist_ok=True)
    out = []
    for label, items in (("tracked", tracked), ("held", held)):
        e_q = emb(items)
        sims = e_q @ e_pool.T
        for r, item in enumerate(items):
            # A tracked item IS in the pool, so its nearest neighbour is itself.
            # That is legitimate: the LoRA also wrote that exact experience. This
            # is the retrieval mechanism's best case and must not be crippled.
            top = np.argsort(-sims[r])[: args.k]
            out.append({
                "split": label,
                "prompt": item["prompt"],
                "completion": item["completion"],
                "tool": item["tool"],
                "neighbors": [
                    {"prompt": pool[j]["prompt"][-EMBED_TAIL:],
                     "completion": pool[j]["completion"],
                     "sim": float(sims[r][j])}
                    for j in top
                ],
            })
        msim = float(np.mean([n["sim"] for o in out if o["split"] == label
                              for n in o["neighbors"]]))
        top1 = float(np.mean([o["neighbors"][0]["sim"] for o in out if o["split"] == label]))
        print(f"  {label:<8} mean top-{args.k} sim {msim:.3f}   mean top-1 sim {top1:.3f}",
              flush=True)

    dest = os.path.join(DATA, "rag", f"{args.name}.jsonl")
    with open(dest, "w") as fh:
        for o in out:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"wrote {dest}  ({len(out)} items)")


if __name__ == "__main__":
    main()
