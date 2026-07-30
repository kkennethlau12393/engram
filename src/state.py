"""A language model whose state lives in its LoRA weights.

Contrast with context memory: a context window is a buffer that is wiped between
sessions, so the model is stateless -- every session re-derives everything from
scratch. Here the LoRA IS the state. It is mutable, it persists to disk, and it
changes while the model runs.

    read(ctx)          -> forward pass. Nothing is injected into the prompt.
    write(ctx, actual) -> gradient step. The experience enters the weights.

Recall is measured as teacher-forced loss on an experience the model was already
written with: one forward pass, no generation, so forgetting can be probed over
hundreds of old memories cheaply.
"""

import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

DEFAULT_MODEL = "mlx-community/Qwen3-4B-4bit"
SYSTEM = (
    "You predict the next tool call a senior engineer would make in this repository. "
    'Reply with one JSON object: {"tool": ..., "input": {...}}. No prose.'
)


class StatefulLM:
    def __init__(
        self,
        model_id=DEFAULT_MODEL,
        rank=8,
        scale=20.0,
        num_layers=8,
        lr=1e-5,
        optimizer="adamw",
        max_len=1536,
    ):
        self.model, self.tokenizer = load(model_id)
        self.model.freeze()
        linear_to_lora_layers(
            self.model,
            num_layers,
            {"rank": rank, "scale": scale, "dropout": 0.0},
        )
        self.model.train()
        self.max_len = max_len
        self.cfg = {
            "model_id": model_id, "rank": rank, "scale": scale,
            "num_layers": num_layers, "lr": lr, "optimizer": optimizer,
        }
        opts = {"adamw": optim.AdamW, "adam": optim.Adam, "sgd": optim.SGD}
        self.opt = opts[optimizer](learning_rate=lr)
        self.loss_and_grad = nn.value_and_grad(self.model, self._loss)
        self.n_writes = 0
        self._init_state = {k: mx.array(v).astype(mx.float32)
                            for k, v in tree_flatten(self.model.trainable_parameters())}

    # ---------- tokenisation ----------

    def encode(self, prompt, completion=None, raw=False):
        """Returns (token_ids, prompt_len). Truncates from the LEFT of the prompt
        so the most recent context and the NEXT_ACTION: marker always survive.

        raw=True skips the chat template and the tool-call system prompt, scoring
        plain text continuation. Required for capability probes: wrapping "The
        capital of France is" in a "reply with one JSON tool call" system prompt
        measures template mismatch, not knowledge."""
        if raw:
            pids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            pids = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, enable_thinking=False
            )
        if completion is None:
            return pids[-self.max_len:], len(pids[-self.max_len:])
        cids = self.tokenizer.encode(completion, add_special_tokens=False)
        room = self.max_len - len(cids)
        if room < 32:  # pathological completion; keep a sliver of prompt
            cids, room = cids[: self.max_len - 32], 32
        pids = pids[-room:]
        return pids + cids, len(pids)

    # ---------- loss ----------

    def _loss(self, tokens, prompt_len):
        """Cross-entropy over completion tokens only."""
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        logits = self.model(inputs)
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = steps >= prompt_len
        ce = nn.losses.cross_entropy(logits, targets) * mask
        ntoks = mask.sum()
        return ce.astype(mx.float32).sum() / ntoks, ntoks

    def score(self, prompt, completion, raw=False):
        """Recall probe: mean per-token loss on this experience. Lower = better
        remembered. No gradient, no generation -- one forward pass."""
        ids, plen = self.encode(prompt, completion, raw=raw)
        tokens = mx.array([ids])
        loss, _ = self._loss(tokens, plen)
        mx.eval(loss)
        return float(loss)

    # ---------- write ----------

    def write(self, prompt, completion, steps=1):
        """Fold one experience into the weights. This is the memory write."""
        ids, plen = self.encode(prompt, completion)
        tokens = mx.array([ids])
        last = None
        for _ in range(steps):
            (loss, _), grads = self.loss_and_grad(tokens, plen)
            self.opt.update(self.model, grads)
            mx.eval(self.model.parameters(), self.opt.state, loss)
            last = float(loss)
        self.n_writes += 1
        return last

    # ---------- read ----------

    def read(self, prompt, max_tokens=48):
        """Generate the next action. Nothing retrieved, nothing injected."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        self.model.eval()
        ids, _ = self.encode(prompt)
        out = generate(
            self.model, self.tokenizer, prompt=ids,
            max_tokens=max_tokens, sampler=make_sampler(temp=0.0), verbose=False,
        )
        self.model.train()
        return out

    # ---------- persistence ----------

    def state_dict(self):
        return {k: v for k, v in tree_flatten(self.model.trainable_parameters())}

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        mx.save_safetensors(path, self.state_dict())
        with open(path.replace(".safetensors", ".json"), "w") as fh:
            json.dump({**self.cfg, "n_writes": self.n_writes}, fh, indent=2)

    def load_state(self, path):
        self.model.update(tree_unflatten(list(mx.load(path).items())))
        mx.eval(self.model.parameters())

    def state_norm(self):
        """Drift: how far state has moved from its INITIAL value. LoRA's A matrices
        are randomly initialised, so an absolute norm is dominated by init and says
        nothing about how much has been written."""
        cur = self.state_dict()
        return float(sum(
            mx.sum((v.astype(mx.float32) - self._init_state[k]) ** 2)
            for k, v in cur.items()
        ) ** 0.5)


def bench(n=5, **kw):
    """Write latency decides whether per-turn statefulness is usable at all."""
    lm = StatefulLM(**kw)
    import glob
    files = sorted(glob.glob(os.path.join(
        os.path.dirname(__file__), "..", "data", "test", "*.jsonl")))
    if not files:
        raise SystemExit("no data/test/*.jsonl -- run extract.py then format.py first")
    rows = [json.loads(l) for l in open(files[0])][: n + 2]
    # First call pays Metal kernel compilation; excluding it or the numbers are
    # warmup, not steady state.
    lm.score(rows[0]["prompt"], rows[0]["completion"])
    lm.write(rows[1]["prompt"], rows[1]["completion"])
    rows = rows[2:]
    t0 = time.time()
    for r in rows:
        lm.score(r["prompt"], r["completion"])
    t_read = (time.time() - t0) / len(rows)
    t0 = time.time()
    for r in rows:
        lm.write(r["prompt"], r["completion"])
    t_write = (time.time() - t0) / len(rows)
    print(f"  trainable params : {sum(v.size for _, v in tree_flatten(lm.model.trainable_parameters())):,}")
    print(f"  score (recall probe): {t_read:.2f}s")
    print(f"  write (1 grad step) : {t_write:.2f}s")
    print(f"  peak mem            : {mx.get_peak_memory() / 1e9:.2f} GB")
    return t_read, t_write


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    if a.bench:
        print(f"{a.model}  rank={a.rank} layers={a.num_layers} max_len={a.max_len}")
        bench(n=a.n, rank=a.rank, num_layers=a.num_layers,
              max_len=a.max_len, model_id=a.model)
