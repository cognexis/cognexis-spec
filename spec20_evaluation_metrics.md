# 20. Evaluation Metrics and Protocols

## 20.1 Purpose

Cognexis evaluation must measure both model quality and computation. A recurrent-depth model can look strong at high loop counts and inefficient at production budgets, or fast at shallow loops and weak on difficult tasks. Evaluation therefore reports performance as a function of loop count, latency, FLOPs, memory, and scheduler behavior.

## 20.2 Evaluation Modes

Required modes:

- Fixed loop count evaluation.
- Sequence-adaptive scheduler evaluation.
- Token-wise scheduler evaluation if implemented.
- Maximum-depth stress evaluation.
- Shallow-depth baseline evaluation.

Fixed loop count evaluation is mandatory because it isolates the effect of recurrent depth from scheduler decisions. Adaptive evaluation is mandatory for any deployment that uses adaptive loops.

## 20.3 Conventional Metrics

Standard language model metrics:

- Perplexity.
- Negative log likelihood.
- Exact match.
- Multiple-choice accuracy.
- Pass@k for code generation.
- BLEU/ROUGE or task-specific generation metrics where appropriate.
- Human preference or rubric scores when automated metrics are insufficient.

Every metric must name the dataset split, prompt format, decoding settings, loop mode, and random seed.

## 20.4 Compute Metrics

Required compute metrics:

- Loops executed per request.
- Mean, median, p90, and p99 latency.
- Prefill latency.
- Decode latency per token.
- Approximate FLOPs.
- Peak memory.
- KV cache memory.
- Throughput in tokens per second.
- Scheduler overhead.

Latency must be measured on specified hardware with batch size, dtype, and backend version recorded. FLOP estimates should distinguish Prelude, recurrent, Coda, and LM-head contributions where possible.

## 20.5 Depth Efficiency Index

Depth Efficiency Index (DEI) measures quality improvement per additional compute between two depths:

```text
DEI(d1, d2) = (metric(d2) - metric(d1)) / (compute(d2) - compute(d1))
```

For metrics where lower is better, such as perplexity or loss:

```text
DEI_loss(d1, d2) = (loss(d1) - loss(d2)) / (compute(d2) - compute(d1))
```

DEI must state the metric and compute unit. Do not compare DEI values across unrelated metrics without normalization.

## 20.6 Loop Saturation Point

The Loop Saturation Point (LSP) is the depth where marginal returns become small or DEI peaks. A practical definition:

```text
LSP = smallest depth d such that DEI(d, next_depth) < threshold
```

or:

```text
LSP = depth with maximum positive DEI
```

Evaluation reports must state which definition is used. LSP should be reported per task because different tasks saturate at different depths.

## 20.7 Overthinking Threshold

The Overthinking Threshold (OT) is the first depth where additional loops harm performance:

```text
OT = smallest d where metric(d_next) < metric(d) - tolerance
```

For loss/perplexity, reverse the comparison. The tolerance prevents noise from being labeled as overthinking. OT should be reported with confidence intervals or bootstrap estimates when possible.

## 20.8 Depth Gain Ratio

Depth Gain Ratio (DGR) summarizes improvement from a shallow baseline to a deeper budget:

```text
DGR = (metric(d_max) - metric(d_min)) / abs(metric(d_min))
```

For lower-is-better metrics:

```text
DGR_loss = (loss(d_min) - loss(d_max)) / loss(d_min)
```

DGR is useful for comparing whether a task benefits from recurrence at all.

## 20.9 Scheduler Metrics

Adaptive scheduler evaluation must report:

- Average loops used.
- Loop distribution histogram.
- Halt reasons.
- Quality at adaptive loops.
- Quality matched to same average fixed loop count.
- False-halt and false-continue rates when oracle traces are available.
- Scheduler overhead.
- Budget violation count.

Adaptive results must be compared with fixed-depth baselines at equal or similar compute. Reporting only adaptive quality without compute comparison is incomplete.

## 20.10 Safety and Reliability Metrics

Evaluation should include:

- Toxicity or harmful-output rates on safety benchmarks.
- Refusal accuracy for disallowed prompts.
- Hallucination or factuality metrics where available.
- Non-finite activation/logit counts.
- Timeout and error rates.
- Compute budget enforcement failures.
- Prompt-injection robustness for scheduler controls.

Depth can affect safety behavior, so safety metrics should be measured across loop counts.

## 20.11 Protocol

Standard protocol:

1. Load checkpoint and resolved config.
2. Verify tokenizer checksum.
3. Select benchmark datasets and prompt templates.
4. Run fixed loop counts over the configured depth grid.
5. Run adaptive scheduler modes.
6. Record quality, compute, memory, and scheduler diagnostics.
7. Compute DEI, LSP, OT, and DGR.
8. Export machine-readable results.
9. Generate summary tables and plots.

Evaluation should avoid changing more than one factor at a time. When comparing with other models, match parameter count, training data where possible, and compute budget.

## 20.12 Result Format

Each result row should include:

```json
{
  "checkpoint": "ckpt-100000",
  "tokenizer_checksum": "sha256:...",
  "dataset": "gsm8k",
  "split": "test",
  "loop_mode": "fixed",
  "loop_count": 8,
  "metric_name": "accuracy",
  "metric_value": 0.0,
  "latency_ms_mean": 0.0,
  "flops_mean": 0.0,
  "hardware": "A100-80GB",
  "dtype": "bf16",
  "seed": 1234
}
```

Use JSONL for row-oriented results and a separate summary JSON for aggregate metrics.

## 20.13 Rust Implementation Notes

The evaluation harness should define:

```rust
pub trait BenchmarkTask {
    fn name(&self) -> &str;
    fn examples(&self) -> Box<dyn Iterator<Item = Example>>;
    fn score(&self, prediction: &Prediction, example: &Example) -> MetricSet;
}
```

The model runner should accept loop options and return diagnostics with predictions. Metrics should be computed in typed structures and serialized at the boundary.

## 20.14 Validation

Evaluation tests must cover:

- Perplexity calculation on a small known dataset.
- Accuracy and exact-match scoring.
- DEI/LSP/OT/DGR formulas.
- Adaptive scheduler diagnostics in result rows.
- Deterministic evaluation with fixed seed.
- Result serialization.
- Failure on tokenizer/checkpoint mismatch.
