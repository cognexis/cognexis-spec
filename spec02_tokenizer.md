# 02. Tokenizer

## 02.1 Purpose

The tokenizer converts raw user-visible text into model-visible token IDs and converts generated token IDs back into text. This component is part of the model contract: changing tokenization changes the input distribution, embedding lookup, vocabulary logits, checkpoint compatibility, and evaluation reproducibility.

Cognexis uses a subword tokenizer, with SentencePiece Unigram or byte-pair encoding as the preferred training algorithms. The tokenizer must operate on Unicode text without requiring external whitespace segmentation. It must support general natural language text, code, structured prompts, chat roles, and control tokens used by the scheduler or serving layer.

## 02.2 Responsibilities

The tokenizer is responsible for:

- Normalizing text according to the tokenizer artifact.
- Segmenting normalized text into subword pieces.
- Mapping each piece to an integer token ID.
- Adding required special tokens such as BOS and EOS when requested.
- Preserving role/control tokens without accidental splitting.
- Decoding token IDs back into valid text.
- Reporting unknown, invalid, reserved, and padding tokens consistently.

The tokenizer must not apply safety filtering, sampling, prompt rewriting, or scheduler decisions. Those responsibilities belong to higher-level inference and safety modules.

## 02.3 Vocabulary Contract

The default vocabulary size is expected to be near 50,000 tokens, but implementations must treat `vocab_size` as a configuration value loaded from the tokenizer artifact and model config. The following special-token classes are required:

- `bos_token`: marks the beginning of a sequence.
- `eos_token`: marks the end of a sequence or generated response.
- `pad_token`: pads sequences in a batch.
- `unk_token`: represents unrecognized pieces if the tokenizer algorithm needs one.
- `eod_token`: separates documents during packed pretraining data.

Instruction-tuned deployments should additionally reserve chat role tokens such as:

```text
<|system|>
<|user|>
<|assistant|>
<|tool|>
<|end|>
```

Special-token IDs must be stable once training begins. Adding tokens after pretraining changes the embedding and LM-head matrix shapes. If new tokens are required, the model must either resize and continue training or use a documented adapter strategy.

## 02.4 Normalization

Tokenizer normalization must be deterministic. The tokenizer artifact must record:

- Unicode normalization form, if any.
- Lowercasing behavior, if any.
- Whitespace handling.
- Byte fallback behavior.
- Treatment of invalid UTF-8 or replacement characters.
- Escape rules for special tokens.

Inference must use the same normalization settings as training. Evaluation harnesses must load the tokenizer from the checkpoint bundle instead of constructing one from defaults.

## 02.5 Encoding Semantics

The canonical encoding API is:

```rust
pub struct EncodeOptions {
    pub add_bos: bool,
    pub add_eos: bool,
    pub allow_special: bool,
    pub max_len: Option<usize>,
    pub truncation: TruncationPolicy,
}

pub trait Tokenizer {
    fn encode(&self, text: &str, options: EncodeOptions) -> Result<Vec<u32>>;
    fn decode(&self, ids: &[u32], options: DecodeOptions) -> Result<String>;
}
```

`add_bos` defaults to `true` for standalone prompts and `false` for text fragments appended to an existing context. `add_eos` defaults to `false` for prompts and `true` for supervised training targets that represent complete assistant responses or documents.

The encoder must reject disallowed special tokens when `allow_special` is `false`. This prevents user text from injecting control tokens accidentally. Trusted prompt templates may enable `allow_special` explicitly.

## 02.6 Decoding Semantics

Decoding must invert tokenization as closely as the tokenizer algorithm allows. The decoder must:

- Stop at EOS when `stop_at_eos` is enabled.
- Skip PAD tokens when `skip_padding` is enabled.
- Preserve whitespace and indentation for code generation.
- Avoid displaying internal scheduler or metadata tokens unless explicitly requested.
- Return a deterministic string for every valid token ID sequence.

Invalid token IDs must produce a descriptive error rather than indexing outside vocabulary bounds. In streaming generation, decode should support incremental operation so newly emitted tokens can be converted without repeatedly decoding the full prefix.

## 02.7 Chat Formatting

Instruction-tuned Cognexis models should use a template-driven chat format. A template converts structured messages into a token sequence:

```text
<|system|>
{system_message}<|end|>
<|user|>
{user_message}<|end|>
<|assistant|>
```

The chat renderer must live above the tokenizer but use tokenizer-owned special-token IDs. Training and inference must use the same template version. Template metadata should be stored with checkpoints because even small formatting changes can shift model behavior.

## 02.8 Code and Structured Text Requirements

For code generation, the tokenizer must preserve:

- Newlines.
- Indentation.
- Common operators.
- Braces, brackets, and punctuation.
- String literal delimiters.
- Markdown code fences.

Byte fallback is strongly recommended so rare identifiers, file paths, Unicode symbols, and binary-adjacent text can be represented without large unknown-token regions. The vocabulary should include frequent programming keywords and operators if code generation is a target capability.

## 02.9 Sequence Length and Truncation

The tokenizer may enforce `max_len` before data reaches the model. Truncation policy must be explicit:

- `error`: reject sequences longer than the limit.
- `left`: keep the newest tokens and discard the oldest prefix.
- `right`: keep the prefix and discard the suffix.
- `middle`: keep beginning and end regions with a truncation marker if configured.

Training data should normally use packing rather than naive truncation. Inference should prefer rejecting overlong prompts or applying a documented context-window policy so users understand what context was removed.

## 02.10 Artifact Format

The tokenizer artifact bundle should contain:

- The tokenizer model file.
- A JSON manifest with tokenizer type and version.
- Special-token strings and IDs.
- Normalization settings.
- Checksum of the vocabulary/model file.
- Chat template version, if applicable.

The model config must reference the tokenizer manifest checksum. Loading a checkpoint with a mismatched tokenizer should fail unless an explicit override is provided for research use.

## 02.11 Rust Implementation Notes

The Rust tokenizer module should expose an immutable, thread-safe tokenizer object. Encoding may be CPU-bound during high-throughput serving, so the implementation should avoid global locks and support parallel calls. Vocabulary lookup tables should be stored in compact structures and validated at load time.

Recommended module shape:

```text
src/tokenizer/
  mod.rs
  sentencepiece.rs
  special_tokens.rs
  chat_template.rs
  tests.rs
```

Use `u32` for token IDs at public boundaries unless the tensor backend requires another integer type internally. Validate all conversions to prevent overflow or negative IDs.

## 02.12 Validation

Tokenizer validation must include:

- Round-trip tests for plain text, multilingual text, code, Markdown, and whitespace-only strings.
- Special-token tests with `allow_special` enabled and disabled.
- Determinism tests across repeated runs.
- Compatibility tests that compare known input strings to golden token ID sequences.
- Decode tests for EOS, PAD, unknown, and invalid IDs.
- Serialization tests for manifest load/save.

Tokenizer golden tests are mandatory because tokenizer drift silently invalidates training and evaluation results.
