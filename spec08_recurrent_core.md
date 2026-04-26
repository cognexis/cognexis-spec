# 08. Recurrent Core

## 08.1 Purpose

The recurrent core is the defining Cognexis component. It applies one shared transformer block repeatedly to refine the hidden state. This creates variable effective depth without adding new block parameters for each depth. Loop count becomes a runtime compute decision rather than a fixed architectural constant.

## 08.2 Recurrence Equation

The baseline recurrence is:

```text
h_0 = prelude_output
for t in 0..loop_count:
    h_{t+1} = R(h_t)
```

`R` is a transformer block with shared parameters. A more stable recurrent update may use residual scaling and gating:

```text
u_t = R(h_t)
g_t = sigmoid(G(norm(h_t), t))
h_{t+1} = g_t * u_t + (1 - g_t) * h_t
```

If input injection is enabled:

```text
h_{t+1} = recurrent_update(h_t, h_0, t)
```

The exact update must be defined by configuration and checkpoint metadata.

## 08.3 Inputs and Outputs

Inputs:

- `h0`: Prelude output, shaped `[batch, seq_len, hidden_dim]`.
- `loop_policy`: fixed or adaptive scheduler.
- `max_loops`: hard maximum loop count.
- `min_loops`: optional minimum loop count before halting is allowed.
- `attention_context`: masks, positions, and cache information.

Outputs:

- `h_final`: final recurrent hidden state.
- `loop_stats`: number of loops executed and scheduler diagnostics.
- Optional `states`: intermediate hidden states for training value heads or analysis.

The recurrent core must always execute at least `min_loops`. It must never execute more than `max_loops`.

## 08.4 Parameter Sharing

The recurrent block parameters are shared across all iterations. This includes:

- Attention projections.
- FFN projections.
- Normalization parameters.
- Optional gate parameters, if the gate is part of the recurrent core.
- Optional input-injection projections.

Optimizer updates must aggregate gradient contributions from every unrolled iteration before applying a weight update. Checkpoint files must store one recurrent block, not one copy per possible loop count.

## 08.5 Scheduler Interaction

The recurrent core owns loop execution but delegates halt decisions to the scheduler. The scheduler may observe:

- Loop index.
- Hidden-state norm and delta.
- Logit confidence from a lightweight probe or full LM head.
- Value-head predicted improvement.
- Remaining compute budget.
- Safety or policy signals.

The recurrent core must call the scheduler at defined points. The default sequence-level adaptive flow is:

```text
h = h0
for t in 0..max_loops:
    h_next = R(h)
    stats = measure(h, h_next)
    if t + 1 >= min_loops and scheduler.should_halt(stats):
        return h_next
    h = h_next
return h
```

For token-wise scheduling, the recurrent core must preserve halted token states while continuing active token states.

## 08.6 Stability Requirements

Repeated application can diverge. The recurrent core must support at least two of the following stability controls:

- RMSNorm or LayerNorm before sublayers.
- Residual scaling.
- Learnable or fixed update gates.
- Spectral norm monitoring.
- Gradient clipping in training.
- Hidden-state norm logging.
- Loop curriculum that ramps depth gradually.

Production inference should track hidden-state norm summaries and halt reason counts. Training should fail fast or alert when recurrent activations become non-finite.

## 08.7 Input Injection

Input injection keeps recurrence anchored to the original contextual state. Without it, repeated application can drift toward a generic fixed point or amplify artifacts. Supported injection modes:

- `none`: baseline research mode, not preferred for high loop counts.
- `residual`: `h_t = h_t + beta * h0_projection`.
- `concat_project`: concatenate `h_t` and `h0`, then project back to `hidden_dim`.
- `gate_condition`: compute gates from both `h_t` and `h0`.
- `attention_memory`: allow recurrent attention to attend to a static memory derived from `h0`.

The selected mode must preserve shape and causal masking. If `h0` is used as memory, position and document masks still apply.

## 08.8 Intermediate State Retention

Training the scheduler or value head may require intermediate states and losses at multiple depths. The recurrent core should support a configurable retention policy:

- `final_only`: keep only the final state.
- `all`: return every loop state.
- `selected`: return states at configured loop indices.
- `checkpointed`: discard activations and recompute during backward.

`final_only` is preferred for inference. Training may use `selected` or `checkpointed` to manage memory.

## 08.9 Decode Semantics

During decode, the recurrent core processes newly generated token positions while attending to cached prefix state according to the attention cache policy. Correctness requires clarity about whether recurrent K/V entries are recomputed each loop or cached per loop.

The default correctness-first rule is:

- Recompute recurrent-core projections for each loop because `h_t` changes.
- Append decode-token cache entries only when they correspond to the state expected by future tokens.
- Do not reuse stale K/V values across loop indices unless the recurrence design explicitly anchors them.

Optimized cache schemes are allowed, but they must have tests proving equivalence to the reference recurrence or must be documented as approximate.

## 08.10 Computational Cost

For fixed loop count `N`, recurrent compute is approximately linear in `N`:

```text
total_compute = prelude_compute + N * recurrent_block_compute + coda_compute + lm_head_compute
```

Adaptive scheduling saves compute only if scheduler overhead and masked execution overhead are lower than skipped recurrent work. Benchmarking must report actual wall-clock latency, not only theoretical loop counts.

## 08.11 Rust Implementation Notes

Recommended API:

```rust
pub struct RecurrentCore {
    block: TransformerBlock,
    gate: Option<RecurrentGate>,
    input_injection: InputInjection,
}

pub struct RecurrentOutput {
    pub hidden: Tensor,
    pub loops_executed: LoopCounts,
    pub diagnostics: LoopDiagnostics,
}

impl RecurrentCore {
    pub fn forward(
        &self,
        h0: &Tensor,
        scheduler: &mut dyn LoopScheduler,
        ctx: &mut RecurrentContext,
    ) -> Result<RecurrentOutput>;
}
```

`LoopCounts` should support both scalar sequence-level counts and per-token counts. Diagnostics should include halt reasons, norm summaries, and budget usage.

## 08.12 Validation

Recurrent core tests must include:

- Fixed loop count executes exactly the requested number of loops.
- Adaptive mode respects `min_loops` and `max_loops`.
- Parameters are shared across loops.
- Gradients accumulate into one parameter set.
- Hidden state shape is preserved.
- Halted token states remain unchanged in token-wise mode.
- Non-finite activations are detected.
- Reference and optimized cache paths match for small examples.
