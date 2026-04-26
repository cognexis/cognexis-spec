# 19. Value Head and Confidence Modeling

## 19.1 Purpose

The value head predicts the expected benefit of running another recurrent loop. It gives the adaptive scheduler a learned estimate of marginal utility instead of relying only on hand-tuned thresholds. The value head must be small enough that its overhead is much lower than a recurrent-core iteration.

## 19.2 Prediction Target

The preferred regression target is improvement in validation objective from loop `t` to loop `t+1`:

```text
target_gain_t = loss_t - loss_{t+1}
```

Positive values mean the next loop improved loss. Negative values mean the next loop hurt the objective. Alternative targets:

- Change in task accuracy.
- Change in log probability of the selected answer.
- Binary continue/halt label based on gain threshold.
- Compute-normalized gain.
- Safety-adjusted gain.

The target must match scheduler usage. If the scheduler compares gain to compute cost, training on compute-normalized gain may be appropriate.

## 19.3 Inputs

The value head may consume:

- Current hidden state `h_t`.
- Loop index embedding.
- Hidden-state delta summary.
- Logit entropy or confidence.
- Initial state summary `h_0`.
- Token role or mask information.

For sequence-level scheduling, token states may be pooled. For token-wise scheduling, the value head produces per-token predictions.

## 19.4 Architecture

Default per-token architecture:

```text
x = rms_norm(h_t)
x = concat(x, loop_embedding[t])       # optional
x = linear(x, bottleneck_dim)
x = activation(x)
gain = linear(x, 1)
```

Default sequence-level architecture:

```text
pooled = masked_mean(rms_norm(h_t), non_pad_mask)
features = concat(pooled, loop_embedding[t], scalar_features)
gain = mlp(features)
```

The bottleneck dimension should be small, commonly 64-512 depending on model size. The value head should not contain a full transformer stack in the baseline.

## 19.5 Output Semantics

The value head may output:

- `predicted_gain`: unbounded scalar regression output.
- `continue_logit`: binary classification logit.
- `risk_adjusted_gain`: gain after safety or latency penalty.
- `uncertainty`: optional variance estimate.

The scheduler must know which output type it is consuming. Mixing regression and classification thresholds without calibration is invalid.

## 19.6 Training Data Collection

To train the value head:

1. Run the base model at multiple loop depths.
2. Store hidden-state summaries or selected hidden tensors at each depth.
3. Compute loss or task metric at each depth.
4. Compute target gain between adjacent depths.
5. Train the value head to predict gain from loop state.

For memory efficiency, collection may happen online during fine-tuning with selected loop states rather than storing full hidden tensors to disk.

## 19.7 Loss Functions

Regression:

```text
L_value = huber(predicted_gain, target_gain)
```

Huber loss is preferred over MSE because gain targets can be noisy. MSE is acceptable for controlled experiments.

Classification:

```text
label = target_gain > gain_threshold
L_continue = binary_cross_entropy(continue_logit, label)
```

Combined:

```text
L = L_lm + lambda_value * L_value + lambda_continue * L_continue
```

The value-head loss weight must be tuned so it does not harm language modeling quality.

## 19.8 Calibration

Calibration aligns predictions with observed gains. Required calibration checks:

- Reliability curve of predicted gain versus actual gain.
- Mean absolute calibration error by loop index.
- False-halt rate: cases where predicted low gain but actual gain is high.
- False-continue rate: cases where predicted high gain but actual gain is low.

Calibration methods:

- Temperature scaling for classification logits.
- Linear scaling of regression outputs.
- Isotonic regression on validation predictions.
- Per-task or per-loop threshold adjustment.

Production thresholds should be chosen from validation curves, not guessed.

## 19.9 Confidence Modeling

Confidence signals complement the value head. Useful metrics:

- Maximum softmax probability.
- Logit margin between top two tokens.
- Entropy.
- Perplexity on known validation targets.
- Self-consistency across loops.

Raw softmax confidence can be miscalibrated. If used for halting, it should be calibrated or combined with value-head gain and hard loop bounds.

## 19.10 Token-Wise Value Head

For token-wise scheduling, the value head predicts:

```text
predicted_gain: [batch, seq_len]
```

It should respect masks:

- PAD tokens have no gain and should halt.
- Prompt tokens may use different thresholds than generated target tokens.
- Special tokens may use conservative thresholds.

Pooling is not appropriate when per-token decisions are required, though sequence-level summary features can be added as context.

## 19.11 Runtime Cost

The value head should add negligible overhead. Requirements:

- Avoid full-vocabulary logits unless already needed.
- Use small MLPs.
- Reuse normalized hidden states when possible.
- Batch value-head calls across tokens and requests.
- Avoid device-to-host synchronization for every loop.

If value-head overhead exceeds saved recurrent compute, the scheduler should fall back to rule-based or fixed mode for that deployment.

## 19.12 Safety-Aware Gain

The value head may include an auxiliary risk prediction:

```text
effective_gain = quality_gain - risk_weight * predicted_risk - latency_weight * cost
```

Safety-risk prediction must be validated carefully and should not replace external safety filters. It can help prevent extra loops from reinforcing harmful or unstable outputs.

## 19.13 Rust Implementation Notes

Recommended API:

```rust
pub struct ValueHead {
    norm: Norm,
    loop_embedding: Option<Embedding>,
    mlp: SmallMlp,
    output: ValueOutputKind,
}

impl ValueHead {
    pub fn predict(&self, hidden: &Tensor, features: &ValueFeatures) -> Result<ValuePrediction>;
}
```

`ValueFeatures` should contain masks, loop index, scalar confidence features, and pooling mode. The value head should be serializable as part of the scheduler or model checkpoint.

## 19.14 Validation

Value-head tests must cover:

- Output shape for sequence-level and token-wise modes.
- Masked pooling correctness.
- Loop embedding bounds.
- Loss computation for regression and classification.
- Calibration report generation.
- Scheduler integration with mocked predictions.
- Runtime overhead measurement.
- Checkpoint save/load.
