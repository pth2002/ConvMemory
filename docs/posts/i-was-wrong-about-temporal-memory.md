# I thought ConvMemory worked because of temporal reasoning. Five seeds proved me wrong.

ConvMemory is a small reranker for long-term agent memory. It sits after vector
search and reorders the candidates before the agent reads them. On LoCoMo it
takes Recall@10 from 0.5345 to 0.7798 for about 17 ms a query, where the
cross-encoder that beats it costs about 2 seconds.

The number held up. The story I told about *why* it worked did not.

This is a writeup of how a mechanism claim I believed for months got refuted by
an experiment I designed to confirm it, and what the project looks like
afterwards.

## 1. The hypothesis

Ordinary memory retrieval treats every memory as an independent vector. That
felt like the obvious thing to fix for **conversational** memory, where the
memories are not independent at all — they arrive in order, they refer back to
each other, and the thing you need is often two turns after the thing that
matches your query.

So ConvMemory does not score memories one at a time. It slides a small learned
window over the memory sequence and lets each candidate's score depend on its
neighbors:

```text
memory stream -> sliding windows -> small conv encoder -> per-candidate score
```

The claim I wrote down: **ConvMemory works because it exploits temporal
structure that a bi-encoder throws away.**

It is a nice story. It explains the architecture, it explains why the gains show
up on conversation data and not on document retrieval, and it gives the project
a reason to exist beyond "I trained a reranker."

## 2. Why the first experiments convinced me

Three results, in order of how convincing they felt at the time.

**The gain was large and stable.** Five split seeds, conversation-level splits,
raw dense top-500 candidate pool:

| Method | Recall@10 | MRR |
|---|---:|---:|
| raw_dense | 0.5345 ± 0.0210 | 0.3254 |
| ConvMemory | 0.7798 ± 0.0074 | 0.5824 |

**A tuned non-neural baseline could not close the gap.** This was the check I
was most worried about — that I had built an expensive way to do BM25 plus a
time-decay prior. So I tuned exactly that: dense retrieval, BM25/lexical
signals, temporal-neighbor propagation, and time decay, tuned on the dev split.

| Method | Recall@10 | MRR |
|---|---:|---:|
| ConvMemory | 0.7798 ± 0.0074 | 0.5824 |
| tuned dense + lexical + temporal decay | 0.7234 ± 0.0227 | 0.4741 |
| dense + BM25 lexical | 0.6473 ± 0.0103 | 0.4739 |

Paired bootstrap: +0.0578 Recall@10, 95% CI [+0.0488, +0.0662]. The learned
model was genuinely doing something a hand-tuned temporal heuristic could not.

**Ablating the temporal window hurt.** Retrained, three seeds:

| Variant | Recall@10 | Δ vs full |
|---|---:|---:|
| full_control | 0.7474 ± 0.0229 | — |
| no_temporal_w1 | 0.7121 ± 0.0232 | **-0.0353** |
| no_lexical | 0.6584 ± 0.0185 | -0.0890 |

Remove the temporal window, lose 3.5 points of recall. Confirmed, I thought.

Look at what I did there. I had a hypothesis about *why*, and I tested it with
an ablation that only shows *whether*. "Removing the temporal component hurts"
is compatible with the temporal story — and also compatible with "the extra
parameters helped," "the neighborhood smoothing helped," and "any second view of
the candidate helped." I had evidence for a component being load-bearing, and I
read it as evidence for a mechanism being real.

I should also have noticed the row above it. Lexical interaction was worth
**2.5x more** than the temporal window in my own ablation table, and I was
writing a project about temporal structure.

## 3. Designing the experiment that could kill it

The fix is not a bigger ablation. It is a **differential** prediction.

If ConvMemory works because it exploits temporal structure, then the benefit of
the temporal window has to be *larger on questions that need temporal reasoning*
than on questions that do not. That is a claim that can fail.

So I split the evaluation set into slices:

- `T_HOP_auto` — the strongest temporal / multi-hop proxy;
- `T_REQUIRED_auto` — questions where temporal information is required;
- `T_SUP_auto` — questions where temporal information is supporting;
- `HARD_NON_TEMPORAL_auto` and `OTHER` — hard questions with no temporal
  requirement, as controls.

Then: retrain the full model and the no-temporal model on the same five seeds
(7, 11, 23, 31, 47), score the same question units, and take a paired bootstrap
of the difference **within each slice**.

Retrained, not inference-time masked. Masking a feature at inference tests a
crippled model; retraining tests the architecture.

The prediction was explicit before the run: temporal slices > non-temporal
controls.

## 4. The result

| Slice | Δ Recall@10 from the temporal window | 95% CI |
|---|---:|---:|
| ALL | +0.0376 | [+0.0306, +0.0451] |
| `T_HOP_auto` (most temporal) | **+0.0096** | **[-0.0037, +0.0230]** |
| `T_REQUIRED_auto` | +0.0252 | [+0.0139, +0.0363] |
| `T_SUP_auto` | +0.0407 | [+0.0219, +0.0603] |
| `OTHER` | +0.0868 | [+0.0672, +0.1045] |
| `HARD_NON_TEMPORAL_auto` (control) | **+0.0838** | **[+0.0650, +0.1040]** |

Backwards. The temporal window helps **most** on the hard non-temporal control
(+0.0838), and its effect on the most temporal slice is **not statistically
significant** (+0.0096, CI crosses zero).

That is not a weak confirmation. It is the opposite of the prediction. A
component that helps everywhere except where the mechanism says it should help
is not implementing that mechanism.

The honest reading: the window is doing generic neighborhood smoothing and
adding capacity. It gives every candidate a second, context-aware view of
itself. That helps most on hard questions — and "hard" is not the same as
"temporal."

## 5. What is actually load-bearing

Rebuilding the description from the evidence rather than from the architecture
diagram:

- **Lexical interaction between query and memory** — the largest single
  contributor, -0.0890 Recall@10 when removed.
- **Fusion with dense similarity** — the base signal.
- **A small neighborhood window** — real, +0.0376 aggregate, but not temporally
  specific.
- **The DCA/router scalar** — worth nothing. Removing it was +0.0017 ± 0.0020,
  i.e. neutral to slightly positive. It shipped as a documented negative result
  rather than a feature.

So the supported one-liner is: *a lightweight learned reranker over fused dense
and lexical features, with a small neighborhood window.* No temporal reasoning
in it.

Note what survived. Against the tuned heuristic, the retrained full model is
still +0.0199 Recall@10 [+0.0105, +0.0283] and +0.0566 MRR. The engineering
result was never in question. Only the explanation was, and the explanation was
the part I was proud of.

## 6. What I did next

Once "temporal structure" stopped being the answer, the question changed to:
*what actually decides whether the right memory ends up first?*

The v0.51 evidence pointed at token-level interaction between the query and the
candidate's own text. So v2 leaned into exactly that: keep v1 as the cheap
high-recall stage, then run a small token-evidence reranker over **only** the
protected top-10.

| Method | R@10 | MRR | ms/query |
|---|---:|---:|---:|
| ConvMemory v1 | 0.7798 | 0.5824 | 16.8 |
| ConvMemory v1 + v2 | 0.7798 | **0.6560** | 28.6 |
| mxbai-rerank-large-v1, full pool | 0.8080 | 0.6688 | 1960.2 |

+0.0734 MRR [+0.0645, +0.0827], recall unchanged by construction, for 12 ms.

This time I tested the mechanism claim up front instead of after. The v2 claim
is "candidate-specific memory text is doing the work," so the ablations attack
exactly that:

| Arm | MRR |
|---|---:|
| full text | 0.6677 |
| scalar features only | 0.5792 |
| memory text removed | 0.2966 |
| memory text shuffled | 0.2731 |
| text from a random other query | 0.2506 |

All three text corruptions fall **below raw dense** (0.3254). The model is not
riding rank/time/score shortcuts; break the text-query correspondence and it
collapses past the point of being useless. That is what a load-bearing mechanism
looks like when you actually test it.

## 7. What I would tell past me

- **An ablation is not a mechanism test.** "Component X matters" and "the model
  works because of the phenomenon X was designed to capture" are different
  claims. Only the second one needs a differential prediction across slices.
- **Write the prediction down before the run.** The slice table only has teeth
  because "temporal slices should beat non-temporal controls" was fixed in
  advance. Afterwards I could have narrated +0.0376 on ALL as a win.
- **Retrain your ablations.** Masking a feature at inference measures a broken
  model, not the architecture.
- **Read your own tables.** The lexical row was 2.5x the temporal row from the
  beginning, in a project named after temporal convolution.
- **Publishing the negative result costs less than you think.** The reranker
  still works, still ships, and the README is now a description I can defend
  line by line. The version where I quietly dropped the failed experiment and
  kept the story is the version that eventually gets caught by someone else.

The full attribution writeup, including the slice definitions and the bootstrap
protocol, is in [NEGATIVE_RESULTS.md](../NEGATIVE_RESULTS.md). What came after is
in [RESEARCH_TRAJECTORY.md](../RESEARCH_TRAJECTORY.md). The reranker itself is
`pip install convmemory`.
