# 10. Language Modeling Head

## 10.1 Purpose

The language modeling (LM) head maps final hidden states to vocabulary logits. It is the final neural component used for next-token prediction. Cognexis uses a standard linear projection, usually tied to the token embedding matrix.

## 10.2 Inputs and Outputs

Input:

```text
hidden: [batch, seq_len, hidden_dim]
```

Output:

```text
logits: [batch, seq_len, vocab_size]
```

For decode, implementations often compute logits only for the last active token:

```text
hidden_last: [batch, 1, hidden_dim]
logits_last: [batch, 1, vocab_size]
```

Training requires logits aligned with next-token targets unless using a sampled or fused loss that avoids materializing the full logits tensor.

## 10.3 Projection

The baseline projection is:

```text
logits = hidden @ embedding_weight^T
```

where:

```text
embedding_weight: [vocab_size, hidden_dim]
```

If embeddings are untied, the LM head owns:

```text
lm_head_weight: [vocab_size, hidden_dim]
```

Bias is disabled by default. If a vocabulary bias is used, it must be configured and serialized:

```text
lm_head_bias: [vocab_size]
```

## 10.4 Final Normalization

The LM head usually receives normalized hidden states:

```text
h = final_rms_norm(coda_output)
logits = lm_head(h)
```

Final normalization stabilizes logit scale across loop counts. It is especially important for adaptive inference because shallow and deep recurrent outputs may have different activation statistics.

The final norm is a separate model component, but LM-head tests should include it in end-to-end logit checks.

## 10.5 Training Loss

The default training loss is next-token cross entropy:

```text
loss = cross_entropy(logits[:, :-1, :], input_ids[:, 1:])
```

Loss masking must exclude:

- PAD tokens.
- Prompt tokens in supervised instruction tuning when only assistant output should be learned.
- Tokens after EOS when examples are padded.
- Optional packed-document boundary regions if configured.

For large vocabularies, implementations may use fused cross-entropy kernels that combine projection, log-softmax, and loss computation. Such kernels must be numerically equivalent to the reference computation within accepted tolerance.

## 10.6 Inference Sampling Interface

The LM head returns logits; sampling is a separate module. Supported sampling controls include:

- Greedy argmax.
- Temperature scaling.
- Top-k filtering.
- Nucleus/top-p filtering.
- Repetition penalties.
- Stop-token detection.
- Beam search for evaluation or constrained decoding.

Loop scheduling must happen before final logits are sampled for a token. The sampler should receive scheduler diagnostics for logging but should not change recurrent states.

## 10.7 Numerical Stability

Softmax consumers must use stable logit handling:

```text
shifted = logits - max(logits)
probs = exp(shifted) / sum(exp(shifted))
```

The LM head itself may output large positive or negative values. Non-finite logits must be treated as an error in training and as a failed generation step in inference. Production serving should record the failure and return a controlled error rather than emitting arbitrary tokens.

## 10.8 Weight Tying

When weight tying is enabled:

- The embedding table and LM-head projection share storage.
- Checkpoint serialization records one tensor.
- Optimizer updates apply once.
- Vocabulary resizing updates both input and output paths.

Weight tying should be validated after checkpoint load by checking parameter identity when the backend supports aliases, or by checking synchronized tensor metadata when aliases are unavailable.

## 10.9 Logit Postprocessing

Before sampling, logits may be postprocessed by:

- Temperature division.
- Token bans for safety or formatting.
- Repetition penalties.
- Minimum-length EOS suppression.
- Grammar constraints for structured output.

These operations belong to the generation module, not the LM head. The LM head must remain a pure projection so evaluation metrics are reproducible.

## 10.10 Rust Implementation Notes

Recommended API:

```rust
pub struct LmHead {
    weight: SharedTensor,
    bias: Option<Tensor>,
    tied_to_embeddings: bool,
}

impl LmHead {
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor>;
    pub fn logits_last(&self, hidden: &Tensor) -> Result<Tensor>;
}
```

For training, expose a fused `loss` path if the backend supports it:

```rust
pub fn cross_entropy_loss(
    &self,
    hidden: &Tensor,
    targets: &Tensor,
    mask: &Tensor,
) -> Result<Tensor>;
```

The fused path must produce the same scalar loss as materializing logits for small reference tests.

## 10.11 Validation

LM-head tests must include:

- Logit shape for full-sequence and last-token paths.
- Weight tying with embeddings.
- Untied mode shape and serialization.
- Cross-entropy masking.
- Stable softmax behavior on large logits.
- Fused loss equivalence to reference loss.
- Decode-time last-token logits match slicing full logits.
- Non-finite logit detection.
