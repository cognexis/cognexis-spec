# 07. Prelude

## 07.1 Purpose

The Prelude is the initial stack of conventional transformer blocks. It prepares token representations before recurrence begins. Its job is to build a stable, contextual hidden state that the shared recurrent core can refine over multiple iterations.

Without a sufficiently capable Prelude, the recurrent core must simultaneously perform early lexical/contextual processing and deeper iterative refinement. That increases training instability and weakens depth generalization. The Prelude therefore acts as the model's front-end feature builder.

## 07.2 Inputs and Outputs

Input:

```text
embedded_hidden: [batch, seq_len, hidden_dim]
```

Output:

```text
h0: [batch, seq_len, hidden_dim]
```

`h0` is the initial recurrent state. It must be preserved or reconstructable when the recurrent core uses input injection.

## 07.3 Layer Configuration

The Prelude contains `num_prelude_blocks` transformer blocks with unique parameters. Typical values are model-size dependent, but all implementations must allow the value to be configured. A small research model may use 1-4 Prelude blocks; larger models may use 6-16 or more.

The Prelude block config includes:

- Attention variant and head counts.
- FFN activation and intermediate dimension.
- Normalization type and epsilon.
- Dropout rates.
- Residual scaling.
- RoPE configuration inherited by attention.

The Prelude may use the same block hyperparameters as the Coda, but it does not have to. Divergent settings must be explicit in the config.

## 07.4 Functional Requirements

The Prelude must:

- Apply each block exactly once in configured order.
- Use causal masking for all self-attention.
- Respect padding and packed-document masks.
- Produce a hidden state compatible with the recurrent core.
- Populate or update KV cache entries when running in prefill/decode.
- Remain deterministic during evaluation and inference.

The Prelude must not perform recurrent looping or adaptive halting. Those behaviors begin only after Prelude output is available.

## 07.5 Role in Training

During pretraining, the Prelude learns general token and context features. Because recurrent loop counts may vary by batch, the Prelude must produce states that remain useful at shallow and deep recurrent budgets. Curriculum training should therefore sample loop counts early enough that the Prelude is exposed to multiple downstream depths.

If the Prelude is too shallow, recurrent iterations may spend capacity repairing underdeveloped representations. If it is too deep, the model spends substantial compute before adaptive scheduling can make any decision. The recommended design process is to tune Prelude depth jointly with expected loop budget and latency target.

## 07.6 Interaction With Input Injection

The recurrent core may inject the Prelude output `h0` into every loop to anchor recurrence to the original contextual state. The Prelude implementation should therefore expose `h0` to the recurrent core without unnecessary copies. If gradient checkpointing is enabled, `h0` must be recoverable during backward computation.

Input injection strategies include:

- Residual injection: add a scaled projection of `h0` each loop.
- Attention injection: include `h0` as an auxiliary key/value source.
- Gate conditioning: compute recurrent gates from both `h_t` and `h0`.

The Prelude does not choose the strategy, but it must provide the state needed by the recurrent core.

## 07.7 KV Cache Behavior

During prefill, Prelude blocks process the full prompt and may store keys/values for later decode. During decode, each Prelude block processes only the new token and appends its K/V entries to the cache. The cache must be indexed by Prelude layer ID separately from Coda and recurrent layers.

The Prelude is not rerun over the full prefix during decode. Cache equivalence tests must prove that incremental decode matches full-sequence execution for small examples.

## 07.8 Stochastic Depth

Prelude blocks may use stochastic depth during training. When enabled, a block can be skipped with a configured probability:

```text
hidden_next = hidden if dropped else block(hidden)
```

Stochastic depth must be disabled for evaluation and inference. Skip probabilities should be lower in early Prelude layers and may increase in deeper layers. The training log should record aggregate drop rates.

## 07.9 Rust Implementation Notes

Recommended API:

```rust
pub struct Prelude {
    blocks: Vec<TransformerBlock>,
}

impl Prelude {
    pub fn forward(&self, hidden: &Tensor, ctx: &mut StageContext) -> Result<Tensor>;
}
```

`StageContext` should provide per-layer block contexts, cache handles, masks, positions, RNG state, and instrumentation hooks. The Prelude should not own the tokenizer or sampler.

## 07.10 Validation

Prelude tests must cover:

- Correct number and order of blocks.
- Shape preservation.
- Unique parameters per block.
- Causal masking.
- Cache equivalence between prefill/decode and full forward.
- Stochastic depth disabled in eval mode.
- Gradient flow from recurrent/coda losses back through Prelude.
- Serialization with stable layer names.
