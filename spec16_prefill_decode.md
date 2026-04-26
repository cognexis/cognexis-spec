# 16. Prefill and Decode Phases

## 16.1 Purpose

Autoregressive inference has two execution phases: prefill and decode. Prefill processes the entire prompt in parallel and initializes caches. Decode generates new tokens one step at a time using cached context. Cognexis must apply recurrent loops correctly in both phases while respecting scheduler decisions, causal masks, and compute budgets.

## 16.2 Prefill Phase

Prefill input:

```text
input_ids: [batch, prompt_len]
```

Prefill operations:

1. Embed all prompt tokens.
2. Run Prelude over the full prompt with causal masking.
3. Run the recurrent core for fixed or adaptive loops.
4. Run Coda over the final recurrent state.
5. Apply final norm and LM head to produce prompt logits if needed.
6. Populate KV caches required for decode.

Prefill is parallel across prompt positions. It is usually compute-heavy for long prompts because attention cost scales with sequence length.

## 16.3 Decode Phase

Decode input at each step:

```text
new_token_id: [batch, 1]
cache_state: previous keys/values and sequence lengths
```

Decode operations:

1. Embed the new token with the correct position ID.
2. Run Prelude for the new token while attending to cached Prelude keys/values.
3. Run recurrent loops for the new token/state according to scheduler policy.
4. Run Coda for the new token while attending to cached Coda keys/values.
5. Compute logits for the new token.
6. Sample or select the next token.
7. Append cache entries and update sequence lengths.

Decode cost grows approximately linearly with generated length and recurrent loop count.

## 16.4 Cache Ownership

The cache must distinguish:

- Prelude layer cache entries.
- Recurrent core cache entries.
- Coda layer cache entries.
- Batch slot and request ownership.
- Sequence length per slot.

A cache entry is valid only for the layer and hidden-state semantics that produced it. Recurrent-core cache reuse requires special care because hidden states change with loop index.

## 16.5 Recurrent Cache Policy

Baseline policy:

- Prelude and Coda caches behave like standard transformer decode caches.
- Recurrent core recomputes loop-dependent projections as needed.
- Any recurrent cache must include loop-index semantics or represent a documented anchored-state approximation.

A safe implementation can choose not to persist recurrent-loop K/V across decode steps initially. This may cost more compute but avoids incorrect reuse of stale states. Optimized recurrent caching must be validated against a reference full-context execution.

## 16.6 Position IDs

Position IDs during prefill are usually:

```text
0, 1, 2, ..., prompt_len - 1
```

During decode:

```text
position = prompt_len + generated_tokens_so_far
```

For left-padded batches, position IDs must ignore padding and count real tokens. For packed training-style inference, position resets must follow the configured document-boundary policy.

## 16.7 Batch Decode

Batch decode must handle sequences with different prompt lengths and finish times. Requirements:

- Use per-slot sequence lengths.
- Mask PAD/cache gaps.
- Stop generating for slots that emitted EOS or hit limits.
- Compact batches when beneficial.
- Preserve request-to-slot mapping for streaming outputs.
- Track per-request loop counts and halt reasons.

Adaptive scheduling may produce different loop counts per batch slot. Dense implementations can run to the maximum active loop count and mask halted slots. Sparse implementations may compact active slots each loop.

## 16.8 Streaming Generation

Streaming generation should emit decoded text as soon as tokens are available. The runtime must:

- Keep tokenizer decode state per request.
- Preserve UTF-8 boundaries.
- Send final stop reason.
- Include optional loop diagnostics if requested.
- Avoid exposing internal hidden states.

Scheduler decisions happen before each emitted token. A token should not be streamed until safety checks and stop-token handling for that token are complete.

## 16.9 Compute Budgets

Every generation request may have:

- Maximum new tokens.
- Maximum loops per token.
- Total recurrent loop budget.
- Wall-clock deadline.
- Memory budget.

The inference engine must enforce hard limits even if the adaptive scheduler predicts further improvement. If budget is exhausted, the generation should stop with a clear reason or fall back to the minimum safe computation path configured by serving policy.

## 16.10 Sampling Controls

Sampling occurs after the LM head. Supported controls:

- Greedy.
- Temperature.
- Top-k.
- Top-p.
- Repetition penalty.
- Stop strings or stop tokens.
- Grammar constraints.

Sampling controls must not modify cache state until a token is selected. If a sampled token is rejected by a safety or grammar layer, the runtime must define whether it resamples, masks the token, or stops.

## 16.11 Error Handling

Inference must handle:

- Prompt longer than context window.
- Cache allocation failure.
- Non-finite logits.
- Invalid token IDs.
- Scheduler errors.
- Request cancellation.
- Backend timeout.

Errors should release cache pages and batch slots. In streaming mode, the client should receive a structured terminal error event.

## 16.12 Rust Implementation Notes

Recommended API:

```rust
pub struct GenerationRequest {
    pub input_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub loop_options: LoopOptions,
    pub sampling: SamplingOptions,
}

pub struct GenerationStepOutput {
    pub token_id: u32,
    pub text_delta: String,
    pub loop_count: usize,
    pub stop_reason: Option<StopReason>,
}
```

Separate `prefill()` and `decode_step()` methods make testing easier. The serving layer can compose them into blocking or streaming APIs.

## 16.13 Validation

Prefill/decode tests must include:

- Full-sequence logits match incremental decode for small examples.
- Position IDs with padding and variable prompt lengths.
- Cache growth and release.
- Fixed loop count in prefill and decode.
- Adaptive loop budget enforcement.
- EOS stopping.
- Streaming UTF-8 correctness.
- Request cancellation cleanup.
- Non-finite logit error handling.
