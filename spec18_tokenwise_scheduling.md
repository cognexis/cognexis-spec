# 18. Token-Wise Loop Allocation

## 18.1 Purpose

Token-wise loop allocation lets different token positions receive different numbers of recurrent iterations. Some positions may already be stable after a few loops, while others may need deeper refinement. This can improve compute efficiency for long prompts and generation workloads with heterogeneous difficulty.

Token-wise scheduling is optional and more complex than sequence-level scheduling. It should be implemented only after fixed and sequence-adaptive modes are correct and well tested.

## 18.2 Conceptual Model

At loop `t`, each token position has an active flag:

```text
active[b, i] = true if token i in batch b should receive another recurrent update
```

The recurrent core updates active positions and preserves halted positions:

```text
h_next = R(h_current)
h_current[b, i] = h_next[b, i] if active[b, i] else h_current[b, i]
```

All positions still exist in the hidden tensor. Halting changes whether their state is updated, not whether they are visible to later attention.

## 18.3 Per-Token Signals

Per-token scheduler signals include:

- Norm of `h_t[i] - h_{t-1}[i]`.
- Relative norm delta.
- Token-level entropy or confidence.
- Value-head predicted gain per token.
- Attention concentration or uncertainty.
- Whether the token is PAD, prompt, generated, or special.
- Task-specific token role, such as assistant target token.

PAD tokens should halt immediately or be masked out entirely. Special tokens may use conservative policies because they structure generation.

## 18.4 Halting Criteria

A token may halt when:

```text
t >= min_loops
and (
  delta_i < min_delta
  or predicted_gain_i < gain_threshold
  or confidence_i > confidence_threshold
  or token_budget_i exhausted
)
```

All tokens halt when `t == max_loops`. The recurrent loop ends when no active non-PAD tokens remain.

The scheduler must prevent a token from reactivating after halting unless an experimental policy explicitly allows reactivation. Baseline token-wise scheduling is monotonic: active can change from true to false, not false to true.

## 18.5 Attention With Mixed-Depth Tokens

A key design question is how active tokens attend to halted tokens. The baseline rule:

- Halted tokens keep their latest hidden state.
- Active tokens may attend to halted tokens using those frozen states.
- Halted tokens do not update even if active tokens change.

This produces a mixed-depth hidden sequence. It is simple and deterministic. More complex designs that update K/V memory for active tokens only must define exact cache semantics and pass reference tests.

## 18.6 Masks and Shapes

The active-token mask has shape:

```text
active_mask: [batch, seq_len]
```

It must combine with:

- Causal mask.
- Padding mask.
- Packed-document boundary mask.
- Finished-sequence mask during generation.

The active mask controls recurrent updates. It must not accidentally remove halted tokens from attention unless the policy explicitly chooses that behavior.

## 18.7 Dense Implementation

The simplest implementation computes the recurrent block for all tokens and applies a mask:

```text
candidate = R(hidden)
hidden = where(active_mask, candidate, hidden)
```

Advantages:

- Simple.
- Uses dense optimized kernels.
- Easy to validate.

Disadvantages:

- Does not save much compute unless kernels can skip inactive tokens.
- Still useful for studying token-wise behavior and collecting metrics.

Dense masked execution is the recommended first implementation.

## 18.8 Sparse or Compacted Implementation

Sparse execution attempts to compute only active token updates. Options:

- Compact active tokens into a smaller batch.
- Use block-sparse attention.
- Use sequence chunking by active spans.
- Skip FFN computation for halted tokens.

Sparse execution must preserve causal attention semantics. It is easy to introduce subtle bugs where active tokens lose access to halted context or cache positions shift incorrectly. Sparse mode requires extensive equivalence tests against dense masked mode.

## 18.9 Decode-Time Token-Wise Scheduling

During standard autoregressive decode, only the newest generated token usually needs recurrent updates. Token-wise scheduling is most useful for:

- Prefill over long prompts.
- Speculative or block generation.
- Reprocessing a chunk of generated tokens.
- Multi-token decoding strategies.

For one-token decode, token-wise scheduling reduces to per-request or per-slot scheduling for the current token. The implementation should still record loop counts per generated token.

## 18.10 Training Considerations

Training token-wise halting requires supervision or regularization:

- Per-token improvement targets from intermediate losses.
- Halting penalty to discourage excessive loops.
- Minimum loop constraints for stable learning.
- Auxiliary loss for value-head calibration.
- Consistency loss so early-halted states remain useful to the Coda.

Initial training should use fixed sequence-level loop counts. Token-wise scheduler training can be introduced after the base model is stable.

## 18.11 Observability

Token-wise scheduling must log:

- Histogram of loops per token.
- Average active tokens per loop.
- Halt reason distribution.
- Token type versus loop count.
- Prompt versus generated token loop counts.
- Quality and latency compared with sequence-level scheduling.

Logs should aggregate by request and avoid raw text unless explicitly enabled with privacy controls.

## 18.12 Failure Modes

Potential failures:

- Important early tokens halt too soon and harm later predictions.
- Mixed-depth states create distribution shift for the Coda.
- Sparse execution corrupts position or cache indices.
- Scheduler learns to halt PAD/punctuation well but fails on semantic tokens.
- Token-wise overhead exceeds saved compute.

Evaluation must compare token-wise scheduling against fixed and sequence-level baselines.

## 18.13 Rust Implementation Notes

Recommended types:

```rust
pub struct TokenLoopState {
    pub active: Tensor,      // bool [batch, seq_len]
    pub loops: Tensor,       // u16 or u32 [batch, seq_len]
    pub halt_reasons: Tensor // enum-coded [batch, seq_len]
}

pub trait TokenScheduler {
    fn observe_tokens(&mut self, obs: TokenSchedulerObservation<'_>) -> Result<TokenDecision>;
}
```

Use compact integer dtypes for loop counts and halt reasons. Keep tensor and host summaries separate to avoid frequent device-to-host synchronization.

## 18.14 Validation

Token-wise scheduling tests must cover:

- Active mask monotonicity.
- Halted token state preservation.
- PAD token handling.
- Dense masked execution correctness.
- Sparse mode equivalence if implemented.
- Loop count histograms.
- Mixed-depth Coda compatibility.
- Decode-time per-token loop logging.
- Budget enforcement per token and per request.
