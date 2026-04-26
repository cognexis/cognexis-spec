# 15. Stability and Normalization

## 15.1 Purpose

Recurrent depth increases the risk of numerical and dynamical instability because the same transformation is applied repeatedly. Small errors can accumulate, hidden norms can drift, and excessive loops can degrade quality. Cognexis therefore requires explicit normalization, residual control, monitoring, and training policies.

## 15.2 Stability Risks

Key risks:

- Hidden-state norm explosion.
- Hidden-state collapse to a low-information fixed point.
- Oscillation between states across loops.
- Attention logit overflow.
- FFN activation overflow in low precision.
- Gradient explosion through unrolled recurrence.
- Quality degradation at loop counts beyond the training distribution.

Stability must be measured, not assumed. Training and evaluation should log hidden norms, deltas, non-finite counts, and per-depth metrics.

## 15.3 Normalization Choices

Cognexis supports:

- `rmsnorm`: default for transformer blocks and final norm.
- `layernorm`: supported for compatibility and experiments.

RMSNorm computes:

```text
rms = sqrt(mean(x^2) + eps)
y = x / rms * weight
```

LayerNorm computes mean-centered normalization:

```text
y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
```

RMSNorm is preferred because it is simple, efficient, and common in modern LLMs.

## 15.4 Norm Placement

Baseline placement:

- Pre-attention norm inside every block.
- Pre-FFN norm inside every block.
- Final RMSNorm after Coda and before LM head.

The recurrent core uses the same norm parameters every loop. Per-loop norm parameters are not baseline Cognexis because they increase parameter count with maximum depth.

## 15.5 Residual Scaling

Residual scaling controls update magnitude:

```text
h_next = h + alpha * update
```

For recurrent blocks, `alpha` may be:

- Fixed scalar, such as `0.5`.
- Depth-scaled scalar, such as `1 / sqrt(max_loops)`.
- Learned scalar initialized below 1.
- Per-channel learned vector.

The default should be conservative for high loop counts. Any learned residual scale must be logged during training because growth above expected bounds can indicate instability.

## 15.6 Gating

Recurrent gating blends candidate updates with the prior state:

```text
candidate = R(h_t)
gate = sigmoid(W_g norm(h_t))
h_{t+1} = gate * candidate + (1 - gate) * h_t
```

Gates help preserve useful information and reduce destructive updates. Gate statistics should be monitored:

- Mean gate value.
- Percent of gates near 0.
- Percent of gates near 1.
- Gate value by loop index.

Collapsed gates may indicate the recurrent block is being ignored or overused.

## 15.7 Spectral Monitoring

The spectral norm or spectral radius of linear maps can indicate whether repeated application may amplify activations. Exact eigenvalue computation is too expensive for large matrices, so use power iteration or approximate singular value tracking.

Monitor at least:

- Recurrent attention output projection.
- Recurrent FFN down projection.
- Optional recurrent gate projection.

Spectral normalization may be applied periodically:

```text
W_normalized = W / max(1, sigma(W) / target_sigma)
```

Do not apply aggressive spectral clipping without evaluating quality; overly contractive updates can prevent useful refinement.

## 15.8 Gradient Controls

Training must support:

- Global gradient norm clipping.
- Optional per-parameter clipping.
- Loss scaling for FP16.
- Non-finite gradient detection.
- Optimizer step skipping on invalid gradients.

Gradient norm should be logged by model stage, especially for recurrent parameters. A recurrent gradient norm that grows with loop count faster than expected may require lower residual scale or curriculum adjustment.

## 15.9 Hidden-State Monitoring

For each loop, collect summary statistics:

- Mean norm.
- Max norm.
- Delta norm `||h_t - h_{t-1}||`.
- Cosine similarity between successive states.
- Non-finite element count.
- Optional entropy or logit confidence.

Production logs should use summaries rather than raw hidden states. Raw activations may leak user data and are too large for routine observability.

## 15.10 Overthinking Detection

More loops can reduce quality after a threshold. Overthinking may appear as:

- Decreasing validation accuracy at high depth.
- Increasing perplexity after a saturation point.
- Repetitive or overconfident generations.
- Hidden-state oscillation.

Evaluation must identify overthinking thresholds per task. The adaptive scheduler should use these findings to avoid harmful extra loops.

## 15.11 Precision Policy

Recommended stability policy:

- Prefer BF16 over FP16 for recurrent training.
- Use stable softmax kernels.
- Keep norm epsilon configurable.
- Accumulate reductions in high precision where available.
- Detect non-finite activations after recurrent loops.

Quantized recurrent-core inference must be validated at multiple loop counts because quantization error can accumulate with depth.

## 15.12 Rust Implementation Notes

Implement normalization as a trait:

```rust
pub trait Normalization {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}
```

Implement stability hooks as optional instrumentation:

```rust
pub trait StabilityObserver {
    fn observe_loop(&mut self, loop_index: usize, hidden: &Tensor, delta: Option<&Tensor>);
}
```

The observer should be lightweight and configurable so it can run in training and selected production deployments.

## 15.13 Validation

Stability tests must include:

- RMSNorm and LayerNorm reference checks.
- Recurrent loop finite-output tests at maximum configured depth.
- Residual scaling behavior.
- Gate output bounds.
- Gradient clipping behavior.
- Non-finite activation detection.
- Spectral monitor sanity checks on small matrices.
- Quantized recurrent loop comparison if quantization is supported.
