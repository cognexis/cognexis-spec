# 05. Feed-Forward Networks

## 05.1 Purpose

The feed-forward network (FFN) is the per-token nonlinear transformation inside each transformer block. Attention mixes information across positions; the FFN transforms each position independently and provides much of the model's representational capacity. In Cognexis, FFNs appear in Prelude blocks, the shared recurrent block, and Coda blocks.

## 05.2 Default Architecture

The default FFN is a SwiGLU-style gated MLP:

```text
gate = activation(X @ W_gate)
up   = X @ W_up
Y    = (gate * up) @ W_down
```

Parameter shapes:

```text
W_gate: [hidden_dim, intermediate_dim]
W_up:   [hidden_dim, intermediate_dim]
W_down: [intermediate_dim, hidden_dim]
```

Bias terms are optional and controlled by configuration. SwiGLU is preferred because it performs well in modern decoder-only language models and gives the recurrent core a controllable nonlinear update.

## 05.3 Alternative Activations

Supported activation modes:

- `swiglu`: default gated SiLU/Swish activation.
- `geglu`: gated GELU activation.
- `gelu`: classic two-layer GELU MLP.
- `relu`: allowed for experiments but not recommended as the default.

For non-gated activations, the FFN computes:

```text
Y = activation(X @ W_1 + b_1) @ W_2 + b_2
```

The config must record activation type because checkpoint weights are not interchangeable across gated and non-gated FFNs.

## 05.4 Intermediate Dimension

`intermediate_dim` controls FFN capacity and cost. Common choices are:

```text
intermediate_dim = 4 * hidden_dim               # classic MLP
intermediate_dim = round_to_multiple(8/3 * hidden_dim, multiple)  # SwiGLU-style
```

The dimension should be rounded to a hardware-friendly multiple such as 64, 128, or 256 depending on tensor core requirements. The rounding policy must be deterministic and stored in the model config.

## 05.5 Inputs and Outputs

Input:

```text
X: [batch, seq_len, hidden_dim]
```

Output:

```text
Y: [batch, seq_len, hidden_dim]
```

The FFN must preserve batch and sequence dimensions. It must not inspect future tokens or mutate attention caches. Token-wise scheduling may pass an active-token mask; inactive tokens should either bypass computation or keep their previous hidden state unchanged in the surrounding recurrent core.

## 05.6 Dropout

Training may apply dropout after the activation/gating product or after `W_down`:

```text
Y = dropout((activation(X @ W_gate) * (X @ W_up)) @ W_down)
```

Dropout must be disabled during evaluation and inference. The dropout rate should be shared with or separately configurable from attention dropout. If both are present, the config must distinguish `attention_dropout` and `ffn_dropout`.

## 05.7 Residual Integration

The FFN itself returns the transformed update. Residual addition belongs to the transformer block:

```text
X_next = X + residual_scale * FFN(norm(X))
```

In the recurrent core, `residual_scale` may be smaller than in Prelude/Coda to stabilize repeated application. The FFN must not secretly apply residual scaling unless it is explicitly part of a fused block kernel.

## 05.8 Precision and Numerical Stability

The FFN may run in BF16/FP16 for speed, but accumulation should use the backend's recommended higher-precision path. Large intermediate activations can overflow in FP16, especially at high recurrent loop counts. Recommended safeguards:

- Use BF16 where available.
- Clamp or monitor activation norms during training.
- Keep normalization parameters in FP32 or backend-recommended precision.
- Use gradient clipping in the training loop.

Quantized inference must define where dequantization occurs. Weight-only quantization is usually safer than activation quantization for the recurrent core because recurrence can amplify quantization error.

## 05.9 Parameter Sharing

Prelude and Coda FFNs use unique weights per block. The recurrent core FFN uses one set of weights shared across every recurrent iteration. Optimizer state for recurrent FFN weights must accumulate gradients from all loop iterations before the update step.

Checkpoint metadata should make shared recurrent parameters explicit. Duplicating recurrent weights per loop in a checkpoint is forbidden for baseline Cognexis because it changes parameter count and defeats the architecture's purpose.

## 05.10 Performance Requirements

The FFN is often the dominant compute cost in transformer blocks. Implementations should:

- Fuse bias, activation, and gating where possible.
- Reuse activation buffers across layers when lifetimes permit.
- Avoid materializing unnecessary transposes.
- Align intermediate dimensions for tensor cores.
- Support gradient checkpointing for training.

For token-wise scheduling, sparse execution may skip halted tokens. Sparse execution should only be enabled when it provides real savings on the target backend; otherwise a dense masked path is simpler and often faster.

## 05.11 Rust Implementation Notes

Recommended API:

```rust
pub struct FeedForwardConfig {
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub activation: Activation,
    pub bias: bool,
    pub dropout: f32,
}

pub struct FeedForward {
    gate_proj: Option<Linear>,
    up_proj: Linear,
    down_proj: Linear,
    activation: Activation,
}
```

The forward method should accept a mode flag or execution context indicating training versus inference. Avoid allocating new tensors for every sub-expression when the backend supports in-place or fused operations safely.

## 05.12 Validation

FFN tests must cover:

- Shape preservation.
- Gated and non-gated activation paths.
- Dropout behavior in training versus eval.
- Deterministic output with fixed weights and no dropout.
- Gradient flow through all parameters.
- Recurrent weight sharing across loop iterations.
- Quantized and mixed-precision load paths if supported.
