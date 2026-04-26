# 03. Embedding Layer

## 03.1 Purpose

The embedding layer maps token IDs into dense vectors of size `hidden_dim`. It is the first differentiable model component after tokenization and establishes the dimensional contract used by every downstream block. Cognexis uses token embeddings plus rotary positional encoding applied inside attention. It does not require learned absolute position embeddings in the default architecture.

## 03.2 Inputs and Outputs

Input:

- `input_ids`: integer tensor shaped `[batch, seq_len]`.
- Optional `position_ids`: integer tensor shaped `[batch, seq_len]`.
- Optional `attention_mask`: mask used later by attention, not by the embedding lookup itself.

Output:

- `hidden`: floating tensor shaped `[batch, seq_len, hidden_dim]`.

Every token ID must be in `[0, vocab_size)`. PAD tokens are embedded like any other token, but later masks must prevent PAD positions from contributing to loss or attention where appropriate.

## 03.3 Parameter Shapes

The required parameter is:

```text
token_embedding: [vocab_size, hidden_dim]
```

If weight tying is enabled, `token_embedding` is shared with the LM head projection. The LM head treats the same matrix as `[vocab_size, hidden_dim]` and computes logits by multiplying hidden states by its transpose.

Optional parameters:

- `embedding_scale`: scalar applied to embeddings before the Prelude.
- `token_type_embedding`: only for specialized multi-segment training; disabled by default.
- `adapter_embedding_delta`: optional low-rank or sparse extension for newly added tokens.

Learned absolute position embeddings are not part of the baseline. If an implementation adds them, the config must mark the checkpoint as nonstandard.

## 03.4 Initialization

Embedding weights should be initialized from a zero-mean distribution with variance matched to `hidden_dim`, for example:

```text
Normal(mean = 0, std = initializer_range)
```

`initializer_range` defaults to `0.02` for small models, but large models may use scaled initialization. Special-token rows should be initialized with the same distribution unless there is a documented reason to zero PAD. If PAD is zeroed, the implementation must ensure weight tying does not force a zero PAD logit row in a way that harms generation.

## 03.5 Positional Information

Cognexis uses rotary positional encoding (RoPE) on query and key vectors inside attention. The embedding layer is responsible for passing or deriving position IDs; the attention layer is responsible for applying rotations.

Position IDs must be monotonically increasing within each sequence after accounting for cached prefix length. During decode, the next generated token receives position:

```text
position = prefill_length + generated_tokens_so_far
```

For packed training sequences, position IDs may either reset after each document boundary or continue across the packed block. The chosen policy must match training and evaluation. Resetting at document boundaries is preferred when the EOD token is used to separate unrelated documents.

## 03.6 Scaling and Normalization

Some transformer implementations scale embeddings by `sqrt(hidden_dim)`. Cognexis should make this behavior explicit through `embedding_scale`. The default is `1.0` unless checkpoint metadata says otherwise.

Embedding output may optionally pass through dropout during training:

```text
hidden = dropout(token_embedding[input_ids] * embedding_scale)
```

Embedding dropout must be disabled during evaluation and inference. Final RMSNorm before the LM head is specified separately and must not be folded into the embedding layer.

## 03.7 Weight Tying Contract

Weight tying is the default. It reduces parameter count and keeps input and output token representations aligned. A conforming tied implementation must ensure:

- The embedding table and LM-head weight refer to the same underlying parameter.
- Optimizer state is stored once or updated consistently.
- Checkpoint serialization does not duplicate or diverge tied weights.
- Resizing the vocabulary updates both views together.

Untied embeddings are allowed only when `tie_embeddings = false` is explicitly configured. Evaluation reports should note when a model uses untied embeddings because parameter counts and behavior differ.

## 03.8 Memory Layout

The preferred memory layout is row-major by token:

```text
offset = token_id * hidden_dim + hidden_index
```

This layout makes lookup contiguous for each token row. For GPU backends, the tensor may be stored in backend-native layout, but the checkpoint format must document the logical shape. Loading code must validate that the number of elements equals `vocab_size * hidden_dim`.

## 03.9 Precision

Training may store embeddings in BF16 or FP16 with FP32 optimizer master weights. Inference may load BF16, FP16, FP8, or quantized embeddings if supported by the backend. Quantized embeddings must preserve the logical output shape and must document dequantization behavior before the Prelude.

Special care is required for rare-token rows. Aggressive quantization can degrade rare words, code identifiers, and multilingual text. Evaluation should include rare-token and code-heavy cases when quantized embeddings are used.

## 03.10 Error Handling

The embedding layer must return clear errors for:

- Token IDs outside vocabulary range.
- Mismatched `hidden_dim`.
- Missing or incompatible tied LM-head weight.
- Position IDs longer than the configured RoPE limit when no extrapolation strategy is enabled.
- Unsupported dtype for the active backend.

Silent clipping of token IDs is forbidden.

## 03.11 Rust Implementation Notes

Recommended API:

```rust
pub struct Embedding {
    pub weight: SharedTensor,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub scale: f32,
}

impl Embedding {
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor>;
}
```

The lookup path should batch-gather rows efficiently. CPU fallback may use direct indexing into contiguous memory; GPU execution should call the backend gather operation. Avoid allocating a temporary vector per token.

If the tensor backend supports parameter aliases, use that mechanism for weight tying. If not, checkpoint load/save must enforce equality and optimizer updates must be carefully synchronized.

## 03.12 Validation

Embedding tests must cover:

- Shape correctness for multiple batch and sequence lengths.
- Out-of-range token ID errors.
- Deterministic lookup for repeated token IDs.
- Weight tying with the LM head after optimizer updates.
- Checkpoint round trip for tied and untied modes.
- Position ID generation for prefill and decode.
- Precision conversion behavior for supported dtypes.
