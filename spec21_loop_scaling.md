# 21. Loop Scaling Experiments

## 21.1 Purpose

Loop scaling experiments measure how Cognexis behavior changes as recurrent depth increases. They answer whether additional loops improve quality, where returns saturate, when overthinking begins, and how compute grows in practice.

These experiments are required before enabling adaptive scheduling in production because the scheduler depends on knowing which depths are useful.

## 21.2 Experimental Questions

The loop scaling suite should answer:

- Does quality improve from 1 loop to the trained maximum?
- Which tasks benefit most from deeper recurrence?
- At what depth does each task saturate?
- Does quality degrade beyond a threshold?
- Are hidden-state norms stable across depth?
- Is latency linear, sublinear, or superlinear in loop count on target hardware?
- How does fixed-depth performance compare to adaptive scheduling at equal compute?

## 21.3 Depth Grid

Use a grid that covers shallow, mid, maximum, and extrapolated depths:

```text
[1, 2, 4, 8, 12, 16]
```

The exact grid depends on `max_loops`. Include every trained curriculum milestone when possible. Extrapolated depths beyond training should be clearly labeled and should not be used for production defaults without validation.

## 21.4 Tasks

Include task categories:

- Language modeling validation perplexity.
- Commonsense or multiple-choice reasoning.
- Math and symbolic reasoning.
- Code generation.
- Summarization or long-context tasks.
- Safety and refusal tasks.
- Synthetic algorithmic tasks for controlled depth analysis.

Depth effects are task-dependent. A single aggregate score can hide important saturation or overthinking patterns.

## 21.5 Controlled Variables

Hold constant:

- Checkpoint.
- Tokenizer.
- Prompt template.
- Decoding parameters.
- Max generated tokens.
- Hardware.
- Batch size, unless measuring throughput scaling.
- Random seed.

Only loop count or scheduler mode should vary in the primary experiment.

## 21.6 Measurements

For each depth and task, record:

- Quality metric.
- Loss/perplexity when labels are available.
- Latency mean/p50/p90/p99.
- Throughput.
- FLOP estimate.
- Peak memory.
- Hidden-state norm by loop.
- Hidden delta by loop.
- Attention entropy if available.
- Non-finite count.
- Example outputs for qualitative review.

Qualitative samples should be selected consistently, not cherry-picked after seeing results.

## 21.7 Analysis

Produce:

- Quality versus loop count plots.
- Quality versus compute plots.
- DEI versus loop interval plots.
- Hidden norm and delta curves.
- Latency versus loop count plots.
- LSP and OT tables per task.
- Adaptive scheduler comparison at matched compute.

Use confidence intervals or repeated runs for stochastic generation tasks. For deterministic multiple-choice evaluation, bootstrap over examples to estimate uncertainty.

## 21.8 Overthinking Study

When performance declines at high depth, investigate:

- Hidden norm growth.
- Decreasing hidden deltas with worse logits.
- Entropy collapse or overconfidence.
- Repetition in generated outputs.
- Task categories most affected.
- Relation to spectral norm and residual scale.

Overthinking findings should feed back into scheduler thresholds and maximum loop defaults.

## 21.9 Extrapolation Study

Evaluate depths beyond those common in training only with safeguards:

- Hard timeout.
- Non-finite activation detection.
- Smaller batch sizes if memory is uncertain.
- Separate result labels.
- No production default changes until validated.

Extrapolation can reveal whether the recurrent block learned a stable iterative refinement or merely memorized the training depth range.

## 21.10 Reporting Template

Each loop scaling report should include:

- Checkpoint and config summary.
- Depth grid.
- Dataset/task list.
- Hardware and backend.
- Quality table.
- Compute table.
- DEI/LSP/OT summary.
- Scheduler implications.
- Failure cases and qualitative examples.

Reports should be reproducible from the saved JSONL results and plotting scripts.

## 21.11 Rust Implementation Notes

The experiment runner should accept:

```text
cognexis eval loop-scaling --config eval.yaml --checkpoint ckpt --depths 1,2,4,8,12
```

Internally, it should reuse the generic evaluation harness and iterate over loop options. Avoid special-case metric code in the loop scaling runner.

## 21.12 Validation

Loop scaling tests must cover:

- Depth grid parsing.
- Result grouping by depth.
- DEI computation between adjacent depths.
- Plot data export.
- Detection of overthinking threshold on synthetic data.
- Matched-compute comparison between adaptive and fixed modes.
