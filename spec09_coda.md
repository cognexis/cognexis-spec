# 09. Coda

## 09.1 Purpose

The Coda is the final stack of conventional transformer blocks after the recurrent core. It integrates the refined hidden state, performs final cross-token mixing, and prepares representations for the language modeling head. Where the Prelude builds an initial state for recurrence, the Coda turns the recurrent result into output-ready features.

## 09.2 Inputs and Outputs

Input:

```text
h_recurrent: [batch, seq_len, hidden_dim]
```

Output:

```text
h_output: [batch, seq_len, hidden_dim]
```

The output is passed to final normalization and the LM head.

## 09.3 Layer Configuration

The Coda contains `num_coda_blocks` transformer blocks with unique parameters. It may mirror the Prelude depth or use a smaller/larger depth depending on deployment goals. Fewer Coda blocks reduce latency; more Coda blocks can improve output calibration and final token mixing.

The Coda config must include:

- Number of blocks.
- Attention variant and head counts.
- FFN activation and intermediate dimension.
- Normalization type.
- Residual scaling.
- Dropout rates.
- Cache policy.

## 09.4 Functional Requirements

The Coda must:

- Run exactly once after recurrent looping completes.
- Apply blocks in configured order.
- Preserve causal masking.
- Respect padding and packed-document masks.
- Use unique parameters per Coda block.
- Update decode KV cache consistently.
- Produce hidden states compatible with final normalization and LM head.

The Coda must not make scheduler halt decisions. Adaptive computation has already completed before the Coda begins, except for future extensions that explicitly define Coda skipping.

## 09.5 Relationship to Recurrence

The recurrent core can produce states at different effective depths. The Coda must be robust across those depths. Training should expose the Coda to the same loop-count distribution used by the recurrent core so it learns to interpret both shallow and deep recurrent outputs.

If the Coda overfits to one loop count, generation quality may degrade in adaptive mode. Evaluation should therefore test all supported loop counts, not only the default loop budget.

## 09.6 Decode Behavior

During prefill, the Coda processes the full prompt after recurrent loops finish. During decode, it processes the new token state and attends to cached prior Coda states. The cache must be layer-specific and must not collide with Prelude or recurrent cache entries.

For adaptive recurrence, the Coda sees final states that may have different loop counts per sequence or token. In token-wise mode, the Coda must accept the mixed-depth hidden tensor and process it normally. Coda attention should not need to know the loop count unless instrumentation or an experimental depth embedding is enabled.

## 09.7 Final Normalization

Many decoder-only models apply a final RMSNorm before logits. Cognexis treats final normalization as a model-level component after the Coda rather than a Coda block itself:

```text
h_norm = final_norm(coda(hidden))
logits = lm_head(h_norm)
```

The final norm must be included even if `num_coda_blocks = 0`, unless a checkpoint explicitly disables it.

## 09.8 Optional Coda Reduction

Latency-sensitive deployments may use a smaller Coda or no Coda. If `num_coda_blocks = 0`, the recurrent output flows directly to final norm and LM head. This is valid but must be reported because it changes architecture and may reduce quality.

Experimental adaptive Coda skipping is allowed only when explicitly configured. It must have its own validation because skipping final layers can produce different calibration behavior than changing recurrent loop count.

## 09.9 Rust Implementation Notes

The Coda can share the same container implementation as the Prelude with different layer IDs and parameters:

```rust
pub struct Coda {
    blocks: Vec<TransformerBlock>,
}

impl Coda {
    pub fn forward(&self, hidden: &Tensor, ctx: &mut StageContext) -> Result<Tensor>;
}
```

Layer names should be stable, for example `coda.blocks.0.attention.q_proj`. Stable names are important for checkpoint loading, parameter freezing, and evaluation diagnostics.

## 09.10 Validation

Coda tests must include:

- Correct block count and ordering.
- Unique parameters per block.
- Shape preservation.
- Cache equivalence for full versus incremental decode.
- Behavior when `num_coda_blocks = 0`.
- Compatibility with multiple recurrent loop counts.
- Gradient flow from LM loss through Coda to recurrent and Prelude stages.
