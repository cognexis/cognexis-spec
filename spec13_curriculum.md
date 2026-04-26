# 13. Curriculum and Loop Scheduling

## 13.1 Purpose

The loop curriculum controls which recurrent depths the model sees during training. A model trained only at one loop count may perform poorly at other loop counts and may become unstable when extrapolated. Cognexis therefore trains with a planned progression from shallow, stable loops to a wider distribution of depths.

This document covers training-time loop sampling. Runtime adaptive scheduling is specified separately in the scheduler documents.

## 13.2 Training Objectives

The curriculum should:

- Stabilize early recurrent training.
- Teach the shared recurrent block to improve representations across multiple depths.
- Prevent the Coda and LM head from overfitting to one depth.
- Provide supervision data for adaptive schedulers and value heads.
- Support extrapolation tests beyond the maximum commonly sampled depth.

The curriculum is part of the experiment definition and must be recorded in checkpoints.

## 13.3 Warm-Up Phase

During warm-up, the model trains with a low maximum loop count:

```text
max_sampled_loops = initial_max_loops
```

Typical values are 1-3 loops. Warm-up reduces gradient variance and lets the recurrent block learn a useful transformation before it is unrolled many times. It also gives the Prelude and Coda a stable target distribution.

Warm-up length may be measured in steps, tokens, or epochs. Step-based control is preferred for large training jobs because epochs may be ill-defined on streaming corpora.

## 13.4 Depth Ramp

After warm-up, the maximum sampled loop count increases toward `target_max_loops`. Supported ramp functions:

- `linear`: increases at a constant rate.
- `cosine`: slow start and smooth finish.
- `exponential`: expands depth quickly after initial stabilization.
- `milestone`: increases at configured step boundaries.

Example:

```text
current_max = initial + floor((target - initial) * ramp_progress)
```

The ramp must never exceed the architecture `model.recurrent.max_loops`.

## 13.5 Sampling Distributions

Once a current maximum is selected, the batch loop count is sampled. Supported distributions:

- `uniform`: equal probability over `[min_loops, current_max]`.
- `geometric`: favors shallow depths while still sampling deeper loops.
- `truncated_normal`: concentrates around a moving mean.
- `stratified`: ensures each depth appears a target number of times per window.
- `fixed`: always uses one depth for ablation.

Uniform sampling is easy to reason about but may overemphasize expensive high-depth batches. Geometric sampling often better matches production use, where most requests should stop early.

## 13.6 Multi-Depth Losses

The model can be trained only on the final sampled depth or on multiple intermediate depths. Multi-depth loss computes logits at selected loop states:

```text
loss = sum_i weight_i * CE(lm_head(h_i), targets)
```

Benefits:

- Encourages useful outputs at shallow depths.
- Provides value-head training targets.
- Improves adaptive scheduler behavior.

Costs:

- Additional LM-head computation.
- More activation retention.
- More complex loss weighting.

The default is final-depth loss for pretraining and selected-depth auxiliary loss during scheduler fine-tuning.

## 13.7 Balancing Gradient Contributions

Because recurrent parameters are reused, deeper loop counts contribute gradients through more unrolled applications. This can bias training toward high-depth behavior. Mitigation options:

- Normalize recurrent gradients by loop count.
- Use loss weights that reduce high-depth dominance.
- Stratify depth sampling.
- Clip gradients globally.
- Monitor per-depth validation metrics.

The selected strategy must be documented. If no gradient normalization is used, evaluation should explicitly check shallow-depth quality.

## 13.8 Scheduler Pretraining

Adaptive scheduler training usually proceeds in phases:

1. Train the model with fixed sampled loop counts.
2. Collect intermediate losses and hidden-state deltas across depths.
3. Train the value head or rule thresholds on the collected depth traces.
4. Fine-tune with adaptive scheduling enabled but bounded by conservative limits.
5. Validate quality and latency under fixed and adaptive modes.

Activating an untrained scheduler early can starve deeper loops of gradient signal. The scheduler should not be allowed to halt below `min_loops`.

## 13.9 Extrapolation Samples

A small fraction of batches may use loop counts above the current ramp maximum but within the architectural maximum. This helps detect and reduce instability at high depth:

```text
if rng() < high_depth_fraction:
    loops = sample(current_max + 1, target_max_loops)
```

High-depth samples should be rare early in training and more common after stabilization. Non-finite activations at high depth should trigger alerts and may reduce the current maximum automatically in experimental training loops.

## 13.10 Task-Aware Curricula

Instruction tuning and code generation may use task-aware loop sampling. Examples:

- Simple classification: shallow-biased sampling.
- Math reasoning: deeper-biased sampling.
- Code synthesis: deeper sampling plus execution feedback.
- Summarization: moderate depth with long-context emphasis.

Task-aware depth labels must be metadata, not user-visible hidden prompts unless the prompt format explicitly supports depth directives.

## 13.11 Distributed Training Requirements

In data-parallel training, all ranks in a synchronized step should generally use the same loop count to keep compute balanced and gradient semantics simple. If ranks use different loop counts, the trainer must account for load imbalance and gradient scaling.

The loop sampler state must be checkpointed:

- RNG seed and position.
- Current ramp progress.
- Current maximum loop count.
- Distribution parameters.
- Recent stratification counters, if any.

## 13.12 Rust Implementation Notes

Recommended API:

```rust
pub trait LoopCurriculum {
    fn sample(&mut self, step: u64, rng: &mut Rng) -> LoopSample;
    fn state_dict(&self) -> CurriculumState;
    fn load_state_dict(&mut self, state: CurriculumState) -> Result<()>;
}

pub struct LoopSample {
    pub min_loops: usize,
    pub max_loops: usize,
    pub sampled_loops: usize,
    pub retain_intermediate: bool,
}
```

The training loop should log sampled depth histograms over time.

## 13.13 Validation

Curriculum tests must cover:

- Warm-up depth limits.
- Ramp schedule correctness.
- Sampling distribution bounds.
- Deterministic sampling with fixed seed.
- Checkpoint and resume of sampler state.
- Distributed broadcast of loop samples.
- Multi-depth loss weighting.
- Rejection of invalid `min_loops`/`max_loops` combinations.
