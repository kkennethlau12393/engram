# engram

**A language model whose memory lives in its weights, not its context window.**

A normal LLM is stateless. The context window is a buffer that gets wiped between sessions, so every session re-derives everything from scratch. Retrieval-augmented memory does not remove that — it refills the buffer, paying tokens on every turn.

This repo tests whether putting the memory in the weights instead is better. **At the scale measured here, it is not: retrieval wins on accuracy** (see below). What parametric state does buy is recall at zero context cost, and a scaling curve that does not depend on a retriever's recall. Both results are reported.

`engram` makes the LoRA itself the state:

```
model = base + LoRA_state          # the LoRA IS the memory

loop:
    response = model(input)               # READ  = forward pass, nothing injected
    LoRA_state ← gradient_step(experience) # WRITE = experience enters the weights
```

The adapter is mutable, persists to disk, and changes while the model runs. Close the process, reopen it, the model still knows — because the weights moved, not because something was pasted back into the prompt.

Named for the *engram*: the physical trace a memory leaves in tissue.

---

## Does it actually work?

It works as a mechanism — memory in weights is real, measurable, and behaviour-changing. It loses to retrieval on accuracy at this scale. Everything below is measured; the negative results are the useful part.

### Memory is real

Written experiences beat a matched never-written control by **+0.422 nats/token** (SE 0.067, 95% CI [+0.291, +0.553], t = 6.31), pooled over **3 independent seeds** and 384 written / 384 held items.

The held-out control is what makes this meaningful. Held items are drawn from the same stream but never written, so if written and held improved equally the model would only have learned the output format — which is not memory. Memory is the *gap*.

| | improvement |
|---|---|
| written items | +2.70 nats/token |
| held control | +2.28 — task acquisition |
| **memory (the gap)** | **+0.42 — ~16% of the total** |

So this is **~84% fast task acquisition, ~16% item memory.** Both real. The skill half is the larger half.

### Behaviour changes, not just probabilities

Loss going down is not the same as the model doing the right thing, so this is verified by generation:

| | parseable | tool accuracy |
|---|---|---|
| base model | **0 / 64** | 0% |
| after 400 writes | **64 / 64** | 84.4% written · 71.9% held |

The base model ignores the instruction and writes prose. After 400 gradient writes it emits valid tool calls. A model that could not do the task at all now does it.

### One write is enough

A single gradient step measurably encodes an experience — 16/16 tracked items improved, mean −0.62 nats/token on the item just written. No replay, no multi-epoch training.

### It learns repo conventions, given exposure

| training stream | correct tool-runner on held-out prompts |
|---|---|
| natural (runner in 6/400 writes) | **0 / 15** |
| enriched (runner in 400/400 writes) | **11 / 20** |

The first null was pure data sparsity — you cannot conclude a model failed to learn something it saw six times. With real exposure the convention sticks. It also conditions correctly on `cwd`, emitting the right per-repo file paths.

This is the case where weights *might* have a structural edge: no file anywhere says "this repo uses bun, narrow the test file before running the suite." That knowledge exists only smeared across hundreds of transcripts, so there is no single document to fetch. **Untested against retrieval, though** — the head-to-head below used general held-out prompts, not the convention task. Treat it as a hypothesis, not a finding.

### Capability survives

General knowledge probes, scored as raw text completion outside the write distribution: **1.302 → 1.315** after 800 writes. No lobotomy.

### The defect: cross-repo interference destroys behaviour

Write 400 experiences from repo A, then 400 from unrelated repo B, then test on repo A.

**By loss:** the memory advantage falls 16% and then plateaus flat. Looks benign.

**By generation:** the model is broken. On repo-A prompts it invents tools that do not exist and emits looping queries from repo B's domain.

```
prompt: a repo-A task
pred:   {"tool": "<a domain noun from repo B>", "input": {"record_id": 1017037, ...}}
pred:   {"tool": "WebSearch", "input": {"query": "…<repo-B topic> …<repo-B topic> …"}}
```

It invents tool names that do not exist, lifted from repo B's subject matter, and
loops. Not degraded output — different output.

**The methodological lesson is the most transferable result here: for a stateful system, teacher-forced loss is not a sufficient measure of retention.** It reported 84% survival on a state that was functionally destroyed. Any conclusion drawn from loss alone needs a generation check.

**A replay buffer fixes it.** Interleaving 30% replayed own-repo experiences during
foreign writes (`--replay-frac 0.3`, replay pool excluding the measured probes so
survival cannot be faked):

| after 400 foreign writes | tool accuracy | bashF1 |
|---|---|---|
| replay 0% | **0.094** — collapsed | 0.000 |
| replay 30% | **0.719** — rescued | 0.221 |
| (no interference at all) | 0.844 | 0.214 |

Tool accuracy falls 84% -> 9% without replay, and the loss curve for that same run
reported the memory gap *widening* to +0.219. Loss saw none of an 89% relative
collapse in behaviour.

### Stable operating point

| lr | outcome |
|---|---|
| **1e-4, 1 step** | **stable — use this** |
| 3e-4, 1 step | **diverges.** Loss rises on everything, drift 24 vs 9.4 |
| 1e-4, 3 steps | 5× the compute for +0.06. Not worth it |

The diverged run is instructive: it showed the *largest* apparent memory effect (+0.921) because both conditions had degraded and written ones degraded slightly less. A gap between two broken numbers is not a result — which is exactly why the control exists.

### Retrieval beats this, on quality

The comparison the premise rests on: same 400 memories, same test items, same base
model — once delivered as weights, once as context (k=5, MiniLM retrieval).

```
                 tool accuracy        bash command F1
              tracked   held        tracked   held
base            0.000   0.000         0.000   0.000
parametric      0.844   0.719         0.214   0.155
retrieval       0.875   0.844         0.584   0.473
```

**Retrieval wins on every metric.** Held items are the fair test — those retrieve
different experiences (similarity 0.751), not themselves — and retrieval still
leads 0.844 vs 0.719 on tool choice and **0.473 vs 0.155 on the actual command,
3x better.**

| | quality | context cost | write cost | interference |
|---|---|---|---|---|
| parametric | lower | **0 tokens** | 1.70 s/write | high, needs replay |
| retrieval | **higher** | ~1,700 tok/query | none | none |

Parametric memory's one real edge is that recall is free at inference. It pays
with lower accuracy, a write per interaction, and a failure mode to defend.

Two untested caveats before writing the idea off: this is 400 memories on a 1.7B
model. Retrieval degrades as the corpus grows and retriever recall falls, while
weights have no such scaling term — and at 109K memories the per-query context
cost becomes the binding constraint rather than a rounding error.

---

## Cost

Measured on an M4, 16 GB, Qwen3-1.7B-4bit, rank 8, 8 layers, 512-token context:

| | |
|---|---|
| write (1 gradient step) | **1.70 s** |
| recall probe (forward pass) | 1.03 s |
| peak memory | 2.5 GB |
| state size on disk | 10 MB |

Base model matters a lot. Context length dominates cost, not LoRA depth:

| base | write | peak |
|---|---|---|
| Qwen3-4B | 3.75 s | 4.2 GB |
| Qwen3-1.7B | 1.70 s | 2.5 GB |
| Qwen3-0.6B | 0.73 s | 1.5 GB |

---

## Where the experiences come from

Claude Code writes every session to `~/.claude/projects/**/*.jsonl`. That is a free expert trajectory dataset — a record of what a strong agent actually did next, with quality labels already embedded.

From one developer's two months: **109,498 actions across 6,257 transcripts.**

Two things worth knowing if you parse these yourself, both of which silently corrupt data:

1. **Transcripts are trees, not lists.** Rewinds and branches mean one file holds several divergent histories linked by `parentUuid`. Each action must be paired with its own ancestor chain.
2. **`sessionId` does not identify a trajectory.** Subagent transcripts inherit their parent's `sessionId`, so grouping by it merges thousands of unrelated agents into one stream. Group by file path.

Quality labels come for free: `is_error` on tool results (2,849 occurrences) and user interrupts (588). Note the interrupt marker's `parentUuid` points at an `attachment` node, not the action that was cut off — you have to walk up to the nearest assistant turn holding a `tool_use`.

---

## Usage

Requires Apple Silicon (MLX/Metal).

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**1 — extract experiences from your transcripts**

```bash
.venv/bin/python src/extract.py --stats     # inspect first: project names and counts
.venv/bin/python src/extract.py             # writes data/actions.jsonl
.venv/bin/python src/label.py               # mine error→retry preference pairs
.venv/bin/python src/format.py              # prompt/completion + temporal splits
```

**2 — run a stateful stream**

```bash
.venv/bin/python src/run_stream.py --project <key> --n 400 --lr 1e-4 --save-state
```

Writes every experience into the weights, probing recall against a held-out control. `--project` takes a key from `extract.py --stats`.

**3 — check it generates, not just scores**

```bash
.venv/bin/python src/gen_eval.py --state adapters/<name>.safetensors --max-tokens 256
```

Always do this. Loss alone will mislead you.

**4 — pool seeds into one effect estimate**

```bash
.venv/bin/python src/run_stream.py --project <key> --seed 29 --name run-s29   # repeat per seed
.venv/bin/python src/pool.py --glob "run-s*.json"
```

`pool.py` refuses to pool runs with duplicate seeds, mixed configs, or divergence — see below.

**5 — per-repo conventions** (needs `repos.json`, copy from `repos.example.json`)

```bash
.venv/bin/python src/convention_test.py --state adapters/<name>.safetensors --max-tokens 256
```

**6 — retrieval baseline** (needs `sentence-transformers`, hence a separate interpreter)

```bash
python3 src/retrieve.py --project <key> --name rag        # system python, has torch
.venv/bin/python src/gen_eval.py --rag data/rag/rag.jsonl # mlx venv
```

---

## Layout

| file | role |
|---|---|
| `src/state.py` | **the mechanism.** `StatefulLM`: read / write / score / persist. 190 lines, domain-agnostic — swap in any `(prompt, completion)` stream |
| `src/run_stream.py` | drives a stream, writes every interaction, probes recall, runs interference + replay |
| `src/extract.py` | transcripts → time-ordered experiences (handles the tree and sessionId traps) |
| `src/label.py` | mines error→retry preference pairs from failures already in the logs |
| `src/format.py` | prompt/completion construction, transcript-level temporal splits |
| `src/gen_eval.py` | generation eval + RAG mode. The check that loss cannot give you |
| `src/convention_test.py` | does state hold per-repo conventions and condition on `cwd`? |
| `src/capability.py` | general-knowledge probes, scored as raw text |
| `src/pool.py` | pooled effect estimates with guards |
| `src/analyze.py` | forgetting curves, per-item normalised retention |

Only `state.py` is the idea. Everything else is measurement.

---

## Things that produced wrong answers

Recorded because each one initially looked like a finding.

- **Probe sets too small.** 16 items against a per-item sd of 0.9 gives SE 0.28 — a CI over a nat wide. The first effect estimate swung +0.699 → +0.196 → +0.495 across samples before settling. Use ≥128.
- **Reruns are not replications.** Two runs sharing a seed score the *same* items. Their agreement is determinism. `pool.py` now refuses to pool duplicate seeds.
- **Age buckets confound cohort with decay.** Only early-written items reach old ages, so a raw bucket mean compares different populations. This manufactured a "70% forgetting curve" that did not exist. Normalise per item.
- **A capability probe must not wear the task's clothes.** Scoring "The capital of France is → Paris" through a *tool-call* system prompt gave a baseline loss of 13.5 and measured template mismatch, not knowledge. Raw-text scoring gives 1.30.
- **Truncated generation makes any keyword metric read zero.** Real commands open with a long `cd` and echo banners; the runner appears ~30 tokens in. At `max_tokens=64` a convention test is structurally guaranteed to score 0/20.
- **A similarity filter on mined preference pairs made them worse.** The worst false pair scored 0.129 Jaccard while three genuine fixes scored 0.000–0.051. Every threshold that caught the bad pairs destroyed more good ones. Left unfiltered on purpose.
- **A sparse convention cannot be tested for.** `bun` appeared in 6 of 400 consecutive experiences. The resulting 0/15 measured the sampling window, not the model.

---

## Limitations

- Apple Silicon only (MLX).
- Validated at 400–800 writes on a 1.7B model. Not tested at the full 109K stream.
- Cross-repo interference is unsolved; mitigations are in flight.
- Recall is measured as loss and as tool-level generation accuracy. Exact-command reproduction is ~6% — it picks the right tool reliably, not the right 200-character bash line.
- Developed against one developer's transcripts. The narrow action space (Bash is 55% of everything) flatters the task-acquisition number and would not transfer to an open-ended domain.
