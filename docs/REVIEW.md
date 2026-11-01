# VERITAS code review

Reviewed: 2026-11-01
Files read in full: `ml/models.py`, `ml/train.py`, `ml/data.py`,
`backend/app/services/verifier.py`, `README.md`, `docs/ARCHITECTURE.md`
Verified by: clean clone into `C:\dev`, Python 3.11.9, CPU torch 2.15,
`pytest` 10/10 passing, `ml.train --arch bilstm_attn` reproduced
(test acc 0.9933, macro-F1 0.9934), backend and frontend both running.

## What's good

The design is sound and follows established practice, not guesswork:

- The siamese encoder with `[c ; e ; |c-e| ; c*e]` matching is the standard
  NLI heuristic from InferSent / ESIM-era work (Mou et al. 2016,
  Conneau et al. 2017). Good choice.
- The attention is Luong-style bilinear, scoring evidence tokens against the
  claim vector. That's the right quantity to be visualising.
- `train.py` restores best-dev-F1 weights rather than final-epoch weights,
  early-stops on macro-F1, seeds torch and numpy, clips gradients, and
  logs a confusion matrix plus timing. Better instrumented than most
  coursework projects.
- INSUFFICIENT is never treated as false. The research framing is honest
  and defensible.

## P0 — fixed on branch `Ishu-h`

### 1. `RNNEncoder` feeds padded batches to the LSTM

`nn.LSTM` is called directly on the padded tensor with no
`pack_padded_sequence`. The forward direction survives because
`masked_mean` discards padded positions afterwards. The backward
direction does not: it starts at the end of the padded sequence and
consumes every pad step before reaching the last real token. Pad
embeddings are zeros, but an LSTM cell still updates its state from a
zero input because the gates have biases.

Consequence: a sentence's representation depends on the longest sequence
in its batch. This affects `bilstm` and `bilstm_attn` — the two
architectures the project is built around — and contaminates the
attention weights, which are computed from those states.

Fixed with `pack_padded_sequence` / `pad_packed_sequence`, using
`total_length=x.size(1)` so the attention mask still lines up.

### 2. `reports/<name>` collides across runs

`name = args.name or args.arch`, so re-running an architecture silently
overwrites the previous `metrics.json`. Changed the default to
`f"{args.arch}_s{args.seed}"`.

### 3. No parameter count or seed in the results

Added `seed` and `n_params` to `metrics.json`. Needed for P1 item 1.

### 4. `decide()` ignored the NOT_ENOUGH_INFO class

The filters tested `probs["SUPPORTED"] >= STANCE_THRESHOLD` without
checking that SUPPORTED was the argmax, so the model's own abstention
vote could be discarded. Now requires argmax and threshold.

The CONFLICTING branch also used `sup[0]` and `ref[0]` — the first item
in each list — while the SUPPORTED and REFUTED branches used `max` over
all items. Made consistent.

`default=1.0` on the INSUFFICIENT branch reported full confidence when
no relevant evidence existed at all. Changed to `0.0`.

## P1 — proposed, not yet done

### 1. The architecture comparison isn't capacity-matched

At `hidden=128`: MLP out_dim 128, CNN 129, LSTM 128, **BiLSTM 256**. The
classifier takes `4 * out_dim`, so BiLSTM carries roughly double the
parameters in both encoder and first classifier layer. A BiLSTM-beats-LSTM
result currently can't be separated from a bigger-model-wins result.

Proposed: run `LSTM --hidden 256` against `BiLSTM --hidden 128` as a
control, and print `n_params` as a column in the results table.

### 2. Attention vs mean-pooling is a second confound

The claim is always mean-pooled; for `bilstm_attn` only the evidence
switches to attention-pooling. So `bilstm` → `bilstm_attn` changes the
pooling of one side, not just "adds attention". A `bilstm_maxpool`
variant would separate "relevance weighting helps" from "attention
specifically helps".

Minor: in the attention path, `masked_mean` runs inside the encoder and
its result is immediately overwritten. Dead compute.

### 3. No baselines

Proposed, both on the same test split, written into the same
`metrics.json` shape so they render in Neural Lab:
- majority-class
- TF-IDF + logistic regression (sklearn already a dependency)

Expect the logistic regression to also score very high on the current
sample data. That is itself the finding — it demonstrates the dataset is
too easy, which supports the warning already in the README.

### 4. Single-seed runs

Every comparison is one run per configuration. Differences of one or two
points are within initialisation noise. Proposed: three seeds per
configuration, reported as mean ± std.

### 5. Evaluation leakage in the sample data

`make_sample_data` reports `evidence=137` across 9,800 pairs, so the same
passages appear in train, dev and test. Combined with templated claim
generation, this makes 0.9933 a memorisation score.

Also `Vocab.build(min_freq=1)` keeps every training word, so test
sentences contain almost no OOV tokens — a second source of inflation.

Proposed: a `--split-by-passage` mode (e.g. passages 0–100 train,
101–120 dev, 121–136 test, none crossing), `min_freq=2` on real runs,
and both numbers reported side by side in the writeup.

### 6. Retrieval score is used as a gate, not a weight

`decide()` filters by `MIN_RETRIEVAL` and then treats every surviving
passage as an equal vote. A passage barely over the threshold can flip a
correct SUPPORTED to CONFLICTING. Proposed: weight each vote by its
retrieval score.

### 7. The confidence number isn't comparable across claims

Every branch returns a max over N items, and a max over more draws is
larger on average. A claim with 10 retrieved passages reports higher
confidence than the same claim with 3, purely from retrieval depth.
Proposed: normalise, or rename the field.

### 8. `MIN_RETRIEVAL` and `STANCE_THRESHOLD` are untuned constants

They determine every verdict the system produces and were never swept.
Proposed: a dev-set sweep plotting verdict distribution against threshold.
That's a figure for the report.

## P2 — small

- `predict()` builds display tokens from `tokenize(text)[:MAX_EV]`, while
  the attention is computed over `vocab.encode(...)`. These agree on the
  normal path, but `encode` returns `[UNK]` for text with no matchable
  tokens where `tokenize` returns `[]`, desyncing the highlights. Worth an
  assertion plus a test, given explainability is the headline feature.
- `MAX_CLAIM, MAX_EV = 32, 64` in `verifier.py` duplicates
  `max_claim=32, max_ev=64` in `ClaimEvidenceDataset`. Should be defined
  once in `ml/data.py`. Silent train/serve skew if one is edited.
- `predict()` encodes and runs the claim once per evidence passage. Encode
  once and expand; roughly halves inference cost.
- `train_seconds` includes per-epoch dev evaluation, so the reported
  training time is inflated, and inflated differently per architecture.
- `latency_ms` is measured over batches of 64, making it per-pair
  throughput rather than latency. Rename, or measure at batch size 1.
- CNN kernels (2, 3, 4) with `padding=k//2` give asymmetric padding for the
  even sizes, then `[:, :, :T]` crops one side. Odd kernels (3, 5, 7) are
  conventional.
- `-1e9` as the mask fill value is fine in fp32, overflows in fp16.
  `torch.finfo(x.dtype).min` is portable.
- `torch.load` without `weights_only=True`.
- Unpinned `>=` requirements mean two machines produce two different
  stacks. Worth pinning for a project reporting numbers.
- No LICENSE, no CI. A workflow running `pytest` on each PR is ~10 minutes
  of setup.

## On explainability

Attention weights are a useful visualisation but are not a guaranteed
explanation — see Jain & Wallace, *Attention is not Explanation* (2019)
and Wiegreffe & Pinter's reply. Suggest describing them as "attention
highlights" rather than the reason for the decision, and adding an
occlusion check: remove the top-attended token and see whether the
prediction actually moves. If it does, that's a real result to show.

## On the demo

The system checks claims only against supplied evidence; it never
searches the web. This is the correct design and matches how FEVER works,
but an unprepared demo will produce INSUFFICIENT repeatedly and look
broken. Suggest one line of UI text stating the scope, and three
pre-tested claims that reliably produce SUPPORTED, REFUTED and
CONFLICTING.

## Proposed split

Mine, on `Ishu-h`:
- P0 fixes 1–4 (done)
- baselines
- multi-seed runs and the results table
- passage-level split

Yours:
- `decide()` weighting and threshold sweep
- the demo script and UI text

Open to swapping any of it.