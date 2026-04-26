# 06. Transformer Block

## 06.1 Purpose

A transformer block composes normalization, self-attention, feed-forward transformation, residual connections, and optional recurrent-specific stabilizers. Cognexis uses the same conceptual block type for Prelude, Recurrent Core, and Coda, but the parameter-sharing and stability settings differ by stage.

## 06.2 Block Contract

Input:

```text
hidden: [batch, seq_len, hidden_dim]
```

Output:

```text
hidden_next: [batch, seq_len, hidden_dim]
```

The block must preserve shape. It may update or read KV cache entries through an explicit attention context. It must not modify tokenizer state, scheduler state, sampling state, or safety state directly.

## 06.3 Pre-Norm Structure

Cognexis uses pre-norm blocks by default:

```text
a = attention(norm_attn(x), attention_context)
x = x + attn_residual_scale * dropout(a)
f = ffn(norm_ffn(x))
y = x + ffn_residual_scale * dropout(f)
```

Pre-norm is preferred because it improves gradient flow in deep and recurrent settings. Post-norm variants are not baseline-compatible unless explicitly configured and retrained.

## 06.4 Normalization

Each block contains separate normalization before attention and before FFN:

```text
norm_attn: RMSNorm or LayerNorm
norm_ffn:  RMSNorm or LayerNorm
```

RMSNorm is the default. Epsilon values must be configurable and stored in checkpoints. Normalization parameters must match `hidden_dim`.

For recurrent blocks, normalization behavior must be identical across loop iterations. Do not create per-loop normalization parameters unless implementing a nonstandard untied recurrent variant.

## 06.5 Residual Scaling

Residual scaling controls update magnitude:

```text
x = x + alpha_attn * attention(...)
y = x + alpha_ffn * ffn(...)
```

The default `alpha` is `1.0` for Prelude and Coda. The recurrent core may use smaller values such as `1 / sqrt(max_loops)` or a learned scalar initialized below 1.0. The chosen strategy must be recorded in the config.

Residual scaling is especially important for high loop counts. Without it, repeated application can amplify hidden-state norms and degrade generation.

## 06.6 Optional Gating

The block may expose an update used by a recurrent gate:

```text
candidate = transformer_block_update(x)
gate = sigmoid(gate_proj(norm_gate(x)))
y = gate * candidate + (1 - gate) * x
```

Gating belongs to the recurrent core wrapper unless the implementation fuses it into the block. Prelude and Coda do not use recurrent gating by default.

## 06.7 Attention Context

The block receives an attention context containing:

- Causal mask.
- Padding or packed-document mask.
- Position IDs.
- KV cache handle.
- Cache update policy.
- Training/eval mode.
- Optional active-token mask for token-wise scheduling.

Passing this context explicitly keeps the block reusable across prefill, decode, training, and evaluation. It also prevents hidden global cache state from leaking between requests.

## 06.8 Parameter Ownership

Prelude:

- Owns a vector of distinct blocks.
- Each block has unique attention, FFN, and norm parameters.

Recurrent Core:

- Owns one block instance.
- Reuses the same parameters every loop.
- Accumulates gradients across loop unrolls.

Coda:

- Owns a vector of distinct blocks.
- Runs once after recurrence.

Checkpoint serialization must preserve this ownership model. A recurrent block must not be serialized as `max_loops` separate parameter sets.

## 06.9 Training Behavior

During training, the block must support:

- Dropout.
- Gradient checkpointing.
- Mixed precision.
- Backpropagation through recurrent unrolls.
- Optional stochastic depth for Prelude/Coda or recurrent iterations.

Stochastic depth in the recurrent core should skip an iteration update while preserving loop accounting. The scheduler and logging should still know that the iteration was sampled and skipped.

## 06.10 Inference Behavior

During inference:

- Dropout and stochastic depth are disabled.
- KV cache updates must be deterministic.
- The block may use fused kernels.
- The block must respect active-token masks in token-wise scheduling.
- The block must return errors rather than panicking on shape/cache mismatches.

For decode, each block usually processes one new token per batch slot while attending to cached prior tokens. For prefill, each block processes the full prompt in parallel.

## 06.11 Rust Implementation Notes

Recommended structure:

```rust
pub struct TransformerBlock {
    pub norm_attn: Norm,
    pub attention: AttentionModule,
    pub norm_ffn: Norm,
    pub ffn: FeedForward,
    pub residual: ResidualConfig,
}

impl TransformerBlock {
    pub fn forward(&self, x: &Tensor, ctx: &mut BlockContext) -> Result<Tensor>;
}
```

`BlockContext` should carry layer ID, cache handles, masks, positions, mode, and instrumentation hooks. Layer ID should be stable for logging and cache indexing.

## 06.12 Validation

Transformer block tests must include:

- Shape preservation.
- Pre-norm ordering.
- Attention and FFN residual paths.
- Dropout disabled in eval.
- Cache update behavior in prefill and decode.
- Shared recurrent block gradients across multiple loops.
- Serialization and deserialization of block parameters.
- Numerical equivalence between unfused reference and fused backend for small cases.
