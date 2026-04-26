# 17. Adaptive Loop Scheduler Design

## 17.1 Purpose

The adaptive loop scheduler decides how many recurrent-core iterations to run. Its job is to improve compute efficiency by stopping when additional loops are unlikely to justify their cost while preserving quality and safety constraints. The scheduler is a runtime policy layer around the recurrent core; it does not replace the recurrent transformation itself.

## 17.2 Scheduler Modes

Supported modes:

- `fixed`: run exactly `loop_count` loops.
- `rule_based`: halt using thresholds on hidden-state deltas, confidence, and budgets.
- `value_head`: halt using a learned prediction of marginal improvement.
- `hybrid`: combine rule-based hard stops with value-head predictions.
- `oracle`: research-only mode using ground-truth future losses for analysis.

Production deployments should start with `fixed` or conservative `hybrid` mode. `oracle` must never be used for real inference because it depends on unavailable future information.

## 17.3 Inputs

The scheduler may observe:

- Loop index `t`.
- Minimum and maximum loop bounds.
- Current hidden state `h_t`.
- Previous hidden state `h_{t-1}`.
- Hidden delta norm.
- Logit confidence or entropy.
- Value-head predicted gain.
- Remaining request compute budget.
- Safety risk signals.
- Per-request quality/latency preference.

The scheduler must not inspect target labels during inference. During training or evaluation, target-aware signals may be used only for supervision or analysis, not for deployed halting decisions.

## 17.4 Outputs

The scheduler returns:

```text
decision: continue | halt
halt_reason: max_loops | min_delta | confidence | value_gain | budget | safety | forced
diagnostics: structured metrics
```

For sequence-level scheduling, the decision applies to the whole sequence or batch item. For token-wise scheduling, the decision may be a mask over token positions.

## 17.5 Hard Bounds

Every scheduler must enforce:

```text
if loops_executed < min_loops: continue
if loops_executed >= max_loops: halt
if compute_budget_exhausted: halt
if safety_override: halt_or_abort
```

These hard bounds take precedence over learned predictions. A value head may recommend continuing, but it cannot exceed the configured budget.

## 17.6 Rule-Based Signals

Rule-based scheduling can use:

Hidden-state delta:

```text
delta_t = mean_norm(h_t - h_{t-1})
halt if delta_t < min_delta
```

Relative delta:

```text
relative_delta_t = norm(h_t - h_{t-1}) / (norm(h_{t-1}) + eps)
```

Logit entropy:

```text
entropy = -sum(p_i * log(p_i))
halt if entropy < entropy_threshold
```

Confidence:

```text
confidence = max_token_probability
halt if confidence > confidence_threshold
```

Rules are transparent and cheap but may not generalize across tasks. Thresholds should be tuned on validation data and reported in evaluation.

## 17.7 Value-Head Scheduling

The value head predicts expected improvement from another loop:

```text
predicted_gain = V(h_t, t, optional_features)
```

The scheduler continues when:

```text
predicted_gain / loop_cost >= gain_threshold
```

`loop_cost` may be a constant per loop or a hardware-aware estimate. For token-wise scheduling, `predicted_gain` can be per token.

The value head should be calibrated so predicted gain corresponds to observed validation improvement. Miscalibration can cause premature halting or wasted loops.

## 17.8 Hybrid Policy

The recommended adaptive scheduler is hybrid:

1. Enforce hard bounds.
2. Require minimum loops.
3. Apply safety and budget overrides.
4. Continue if hidden-state delta is still large.
5. Halt if confidence is high and predicted gain is low.
6. Otherwise use value-head gain per cost.

Example:

```text
if t < min_loops:
    continue
if t >= max_loops:
    halt(max_loops)
if budget.remaining_loops == 0:
    halt(budget)
if safety.risk_increasing:
    halt(safety)
if delta < min_delta and predicted_gain < gain_threshold:
    halt(value_gain)
if confidence > confidence_threshold and predicted_gain < high_conf_gain_threshold:
    halt(confidence)
continue
```

## 17.9 Scheduler State

The scheduler should maintain lightweight per-request state:

- Loops executed.
- Previous hidden summary.
- Budget consumed.
- Halt reasons.
- Value-head predictions by loop.
- Confidence and entropy by loop.

It should not store full hidden states unless training or debugging explicitly requests them. Production state must be small enough for high-concurrency serving.

## 17.10 Batch Semantics

In batched inference, each request may have different scheduler decisions. A sequence-level adaptive implementation may:

- Run all batch slots until every slot halts.
- Mask halted slots while active slots continue.
- Compact active slots between loops.

Masking is simpler; compaction may be faster when many slots halt early. The implementation must preserve request ordering and cache ownership either way.

## 17.11 Training the Scheduler

Training data for scheduler decisions can be collected by running fixed-depth passes and recording:

- Loss at each loop.
- Accuracy or task score at each loop.
- Hidden-state delta.
- Confidence/entropy.
- Actual improvement from loop `t` to `t+1`.

The value-head document defines the learned model. This scheduler document defines how predictions are consumed at runtime.

## 17.12 Budget Model

The scheduler should support budgets expressed as:

- Maximum loops per token.
- Maximum total loops per request.
- Approximate recurrent FLOPs.
- Wall-clock deadline.
- User-selected quality tier.

Hardware-aware cost estimates may use moving averages from recent requests. If wall-clock deadline is near, the scheduler must stop even if loop budget remains.

## 17.13 Safety Interaction

Safety policies can influence scheduling:

- Halt if additional loops increase harmful-output risk.
- Force minimum loops for safety-critical classification before generation.
- Cap loops for suspicious prompt-injection attempts that request maximum compute.
- Abort generation if filters require refusal.

Safety overrides must be logged as halt reasons without exposing sensitive prompt content.

## 17.14 Rust Implementation Notes

Recommended trait:

```rust
pub trait LoopScheduler {
    fn begin_request(&mut self, ctx: &SchedulerRequestContext) -> Result<()>;
    fn observe(&mut self, obs: SchedulerObservation<'_>) -> Result<SchedulerDecision>;
    fn finish(&mut self) -> SchedulerDiagnostics;
}

pub struct SchedulerDecision {
    pub action: LoopAction,
    pub halt_reason: Option<HaltReason>,
}
```

`SchedulerObservation` should contain summaries and tensor references as needed. The scheduler should not own model weights except optional value-head parameters.

## 17.15 Validation

Scheduler tests must include:

- Fixed mode exact loop count.
- Hard max loop enforcement.
- Minimum loop enforcement.
- Budget exhaustion.
- Rule threshold behavior on synthetic observations.
- Value-head threshold behavior with mocked predictions.
- Batch slots halting at different loops.
- Structured diagnostics and halt reasons.
- No use of target labels during inference.
