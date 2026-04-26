# 25. Glossary

## 25.1 Purpose

This glossary defines terms used across the Cognexis specification. Terms are written to be implementation-oriented so configuration names, logs, metrics, and code can use the same vocabulary.

## 25.2 Terms

- **Adaptive Loop Scheduler:** Runtime policy that decides whether to run another recurrent-core iteration. It may use rules, confidence signals, a value head, budgets, or safety signals.
- **Attention Head:** One projection subspace in self-attention. Query heads may share key/value heads under grouped query attention.
- **Attention Mask:** Tensor or structured metadata that prevents attention to future tokens, padding tokens, or disallowed packed-document regions.
- **Backpropagation Through Time (BPTT):** Gradient computation through the unrolled recurrent core. Although Cognexis is a transformer-like model, repeated loops create a time-like unroll.
- **BOS Token:** Beginning-of-sequence token inserted before standalone prompts or documents when configured.
- **Causal Attention:** Attention constraint where token position `i` can attend only to positions `j <= i`.
- **Coda:** Stack of conventional transformer blocks applied after the recurrent core and before final normalization/LM head.
- **Compute Budget:** Runtime limit on loops, FLOPs, wall-clock time, memory, or generated tokens.
- **Context Window:** Maximum number of tokens the model can process in one sequence, including prompt and generated tokens.
- **Decode:** Autoregressive generation phase that processes one or more newly generated tokens using cached prior context.
- **Depth Efficiency Index (DEI):** Quality improvement per additional unit of compute between two loop depths.
- **Depth Gain Ratio (DGR):** Relative improvement between shallow and deep loop settings.
- **Document Boundary Mask:** Optional mask that prevents packed documents from attending to each other.
- **Effective Depth:** Number of conceptual block applications: Prelude blocks plus recurrent loops plus Coda blocks.
- **EOS Token:** End-of-sequence token that can stop training examples or generation.
- **EOD Token:** End-of-document token used to separate packed pretraining documents.
- **FFN:** Feed-forward network inside a transformer block, usually a gated MLP in Cognexis.
- **Fixed Loop Mode:** Inference or training mode where the recurrent core runs an exact configured number of loops.
- **Gating:** Mechanism that blends a recurrent candidate update with the previous hidden state.
- **Grouped Query Attention (GQA):** Attention variant where multiple query heads share fewer key/value heads.
- **Halt Reason:** Structured reason reported by a scheduler when recurrent looping stops.
- **Hidden State:** Dense vector representation for each token at a given model stage and loop.
- **Input Injection:** Technique that feeds the Prelude output or original context into recurrent loops to anchor iterative refinement.
- **KV Cache:** Key/value cache used during decode so previous tokens do not need to be recomputed in standard attention layers.
- **Language Modeling Head (LM Head):** Projection from hidden states to vocabulary logits.
- **LayerNorm:** Normalization that subtracts mean and divides by standard deviation over the hidden dimension.
- **Loop Count:** Number of recurrent-core iterations executed.
- **Loop Curriculum:** Training schedule that controls sampled recurrent depths over time.
- **Loop Saturation Point (LSP):** Depth at which additional loops stop providing meaningful returns or DEI peaks.
- **Max Loops:** Hard upper bound on recurrent iterations.
- **Min Loops:** Minimum recurrent iterations before adaptive halting is allowed.
- **Multi-Query Attention (MQA):** Attention variant with many query heads and one shared key/value head.
- **Overthinking Threshold (OT):** First depth where additional loops reduce quality beyond a tolerance.
- **PAD Token:** Padding token used to align batch sequences.
- **Perplexity:** Exponential of average negative log likelihood; lower is better.
- **Position IDs:** Integer positions used to apply RoPE and manage cached decode.
- **Prefill:** Inference phase that processes the full prompt and initializes hidden states/caches.
- **Prelude:** Stack of conventional transformer blocks applied before the recurrent core.
- **Recurrent Core:** Shared transformer block applied repeatedly to refine hidden states.
- **Residual Scaling:** Multiplicative scale on residual updates, used to control recurrent dynamics.
- **RMSNorm:** Normalization that divides by root mean square over the hidden dimension without mean subtraction.
- **RoPE:** Rotary positional encoding applied to attention query and key vectors.
- **Sampler:** Generation component that converts logits into selected next tokens.
- **Sequence-Adaptive Scheduling:** Scheduler mode where a halt decision applies to an entire sequence or batch slot.
- **SFT:** Supervised fine-tuning on instruction/response examples.
- **Special Token:** Reserved token with control semantics, such as BOS, EOS, role markers, or EOD.
- **Spectral Monitoring:** Approximate tracking of matrix spectral norm/radius to detect recurrent instability risk.
- **Stochastic Depth:** Training regularizer that randomly skips blocks or recurrent updates.
- **Token-Wise Scheduling:** Scheduler mode where different token positions may run different numbers of recurrent loops.
- **Value Head:** Small neural head that predicts expected benefit from another recurrent loop.
- **Weight Tying:** Sharing the token embedding matrix with the LM-head projection.

## 25.3 Preferred Metric Names

Use these names in result files:

```text
perplexity
negative_log_likelihood
accuracy
exact_match
pass_at_k
latency_ms
prefill_latency_ms
decode_latency_ms_per_token
flops
peak_memory_bytes
loops_mean
loops_p50
loops_p90
loops_p99
depth_efficiency_index
loop_saturation_point
overthinking_threshold
depth_gain_ratio
budget_exhaustion_rate
```

## 25.4 Preferred Halt Reasons

Use these enum-like names in logs and APIs:

```text
min_loops_not_met
max_loops
min_delta
confidence
value_gain
budget
safety
all_tokens_halted
forced
error
```

## 25.5 Naming Guidance

Configuration, code, and logs should prefer the terms in this glossary. Avoid using multiple names for the same concept, such as `iterations`, `steps`, and `loops`, unless the distinction is explicit. In this specification, `loop` means one recurrent-core application.
