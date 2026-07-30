"""Does continuous writing lobotomise the base model?

A memory system that degrades general ability is not a win, however well it
recalls. These probes are deliberately OUTSIDE the transcript distribution --
general knowledge, arithmetic, code reading, instruction following. Nothing here
resembles a tool call, so writing tool calls should not improve them; any
movement is interference.

Reported as loss delta vs the untouched base model. Positive = degraded.
"""

PROBES = [
    ("The capital of France is", " Paris."),
    ("Water boils at", " 100 degrees Celsius at sea level."),
    ("2 + 2 =", " 4"),
    ("17 * 3 =", " 51"),
    ("The opposite of 'increase' is", " decrease."),
    ("Python list comprehension that squares numbers 1-5:", " [x**2 for x in range(1, 6)]"),
    ("In git, the command to create a new branch is", " git branch <name>"),
    ("A function that returns the larger of two numbers:", " def larger(a, b): return a if a > b else b"),
    ("The largest planet in our solar system is", " Jupiter."),
    ("Translate to French: 'good morning' ->", " bonjour"),
    ("HTTP status code for 'not found' is", " 404"),
    ("The chemical symbol for gold is", " Au"),
    ("Shakespeare wrote", " Hamlet, Macbeth, and many other plays."),
    ("To reverse a string in Python:", " s[::-1]"),
    ("The square root of 144 is", " 12"),
    ("DNA stands for", " deoxyribonucleic acid."),
]


def measure(lm):
    """Mean loss over the probe set, scored as RAW text continuation.

    Must not use the tool-call chat template: wrapping "The capital of France is"
    in a "reply with one JSON tool call" system prompt measures template
    mismatch, not knowledge. That bug made an early run report baseline loss of
    13.5 on trivial facts."""
    return [lm.score(p, c, raw=True) for p, c in PROBES]


def report(before, after):
    import statistics

    b, a = statistics.mean(before), statistics.mean(after)
    print(f"\n  CAPABILITY (general knowledge, outside the write distribution)")
    print(f"    before {b:.3f}  ->  after {a:.3f}   delta {a - b:+.3f}"
          f"   {'DEGRADED' if a - b > 0.15 else 'stable'}")
    worst = sorted(
        ((a_ - b_, PROBES[i][0]) for i, (b_, a_) in enumerate(zip(before, after))),
        reverse=True,
    )[:3]
    for d, prompt in worst:
        print(f"      {d:+.3f}  {prompt[:52]}")
    return a - b
