# 11. Configuration Files

## 11.1 Purpose

Cognexis configuration files define architecture, tokenizer compatibility, training behavior, inference defaults, evaluation protocols, and safety controls. Configuration is part of the checkpoint contract. A model must not be loaded without the configuration required to interpret its tensors and runtime behavior.

The preferred format is YAML for human-authored experiment configs and JSON for serialized manifests. Both formats must deserialize into the same typed Rust structs.

## 11.2 Configuration Principles

Configuration must be:

- Explicit: behavior-changing defaults should be visible in generated resolved configs.
- Validated: invalid combinations must fail at startup with descriptive errors.
- Versioned: every config includes a schema version.
- Reproducible: checkpoints include the resolved config used for training.
- Overrideable: command-line and serving overrides may change runtime fields such as loop budget but not architecture fields.

The loader should distinguish between training-time architecture fields and request-time inference controls. A serving request may set `max_loops`; it may not change `hidden_dim`.

## 11.3 Top-Level Schema

A complete config contains:

```yaml
schema_version: 1
run:
  name: cognexis-small
  seed: 1234
tokenizer: {}
model: {}
training: {}
inference: {}
evaluation: {}
safety: {}
logging: {}
```

Sections may be omitted only when they are irrelevant to the command being run. For example, an evaluation-only command may omit `training`, but a checkpoint manifest must include the model and tokenizer sections.

## 11.4 Tokenizer Configuration

```yaml
tokenizer:
  type: sentencepiece
  path: tokenizer.model
  manifest_path: tokenizer.json
  vocab_size: 50000
  bos_token: "<s>"
  eos_token: "</s>"
  pad_token: "<pad>"
  eod_token: "<eod>"
  allow_byte_fallback: true
  add_bos_default: true
  add_eos_default: false
  chat_template: chatml_v1
  checksum: "sha256:..."
```

Validation requirements:

- `vocab_size` must match the tokenizer artifact and embedding matrix.
- Special-token IDs must be unique.
- The checksum must match unless explicitly bypassed for research.
- The chat template must reference known special tokens.

## 11.5 Model Configuration

```yaml
model:
  dtype: bf16
  vocab_size: 50000
  hidden_dim: 4096
  num_prelude_blocks: 8
  num_recurrent_blocks: 1
  num_coda_blocks: 8
  tie_embeddings: true
  final_norm: rmsnorm
  norm_epsilon: 1.0e-5
  attention:
    variant: gqa
    num_attention_heads: 32
    num_kv_heads: 8
    head_dim: 128
    attention_bias: false
    dropout: 0.0
  rope:
    theta: 10000.0
    max_position_embeddings: 8192
    scaling:
      type: none
  ffn:
    activation: swiglu
    intermediate_dim: 11008
    bias: false
    dropout: 0.0
  recurrent:
    max_loops: 12
    min_loops: 1
    residual_scale: 0.5
    gating: true
    input_injection: residual
    return_intermediate_states: false
```

Validation requirements:

- `hidden_dim == num_attention_heads * head_dim`.
- `num_attention_heads % num_kv_heads == 0`.
- `num_recurrent_blocks == 1` for baseline Cognexis.
- `max_loops >= min_loops >= 1`.
- `vocab_size` matches tokenizer config.
- `intermediate_dim` is positive and compatible with backend alignment constraints.

## 11.6 Training Configuration

```yaml
training:
  data:
    train_manifest: data/train_manifest.jsonl
    validation_manifest: data/valid_manifest.jsonl
    sequence_length: 2048
    packing: true
    document_boundary_attention: false
    num_workers: 16
    prefetch_batches: 8
  optimization:
    optimizer: adamw
    learning_rate: 2.0e-4
    min_learning_rate: 2.0e-5
    weight_decay: 0.1
    beta1: 0.9
    beta2: 0.95
    epsilon: 1.0e-8
    gradient_clip_norm: 1.0
    warmup_steps: 2000
    total_steps: 500000
    scheduler: cosine
  batch:
    global_batch_tokens: 4194304
    micro_batch_size: 2
    gradient_accumulation_steps: 16
  precision:
    compute_dtype: bf16
    optimizer_state_dtype: fp32
    loss_scale: dynamic
  loop_curriculum:
    mode: ramp_then_sample
    warmup_steps: 10000
    initial_max_loops: 2
    target_max_loops: 12
    sampling_distribution: truncated_geometric
    high_depth_fraction: 0.05
  checkpoint:
    save_every_steps: 1000
    keep_last: 5
    keep_best: 3
    output_dir: checkpoints/
```

Training configs must record enough information to reproduce loop sampling and data ordering. Distributed jobs must include rank/world-size information in runtime metadata even if not in the static config.

## 11.7 Inference Configuration

```yaml
inference:
  max_sequence_length: 8192
  max_new_tokens: 512
  loop_mode: adaptive_sequence
  min_loops: 1
  max_loops: 8
  compute_budget:
    max_recurrent_flops: null
    max_wall_time_ms: null
  scheduler:
    type: value_head
    min_delta: 0.0001
    confidence_threshold: 0.95
    predicted_gain_threshold: 0.0005
  sampling:
    temperature: 0.7
    top_p: 0.95
    top_k: 0
    repetition_penalty: 1.0
  cache:
    type: paged
    dtype: bf16
    max_batch_size: 16
```

Serving may override inference fields per request, subject to configured safety and budget limits. Overrides must be validated before generation starts.

## 11.8 Evaluation Configuration

```yaml
evaluation:
  loop_counts: [1, 2, 4, 8, 12]
  adaptive_modes: [fixed, adaptive_sequence]
  datasets:
    - name: validation_ppl
      type: language_modeling
      path: data/valid.jsonl
    - name: gsm8k
      type: exact_match
      path: data/gsm8k.jsonl
  metrics:
    - perplexity
    - accuracy
    - latency_ms
    - flops
    - depth_efficiency_index
    - loop_saturation_point
    - overthinking_threshold
  output_path: eval/results.jsonl
```

Evaluation must report the resolved config, checkpoint ID, tokenizer checksum, hardware summary, dtype, and loop mode.

## 11.9 Safety and Monitoring Configuration

```yaml
safety:
  input_filter: enabled
  output_filter: enabled
  code_sandbox: wasi
  max_user_loops: 12
  block_special_token_injection: true
  pii_logging: redacted
logging:
  level: info
  format: json
  log_loop_stats: true
  log_hidden_norms: summary
  metrics_endpoint: "127.0.0.1:9090"
```

Safety config must be loaded before serving requests. Request overrides must not disable production safety filters unless the process is explicitly launched in a research mode.

## 11.10 Defaults and Resolved Configs

The config loader may provide defaults, but every run should emit a resolved config containing all effective values. This resolved config is what should be written into checkpoints and evaluation outputs.

Defaults should be conservative:

- Fixed or sequence-adaptive loops before token-wise loops.
- `min_loops = 1`.
- `max_loops` no greater than the trained maximum unless research override is enabled.
- Safety filters enabled in serving.
- Dropout disabled outside training.

## 11.11 Error Handling

Config loading must produce structured errors for:

- Unknown required fields.
- Invalid enum values.
- Shape-incompatible dimensions.
- Missing tokenizer or checkpoint artifacts.
- Unsafe serving overrides.
- Training configs that cannot produce a valid loop curriculum.

Warnings are appropriate for deprecated fields, nonstandard experimental fields, or evaluation beyond trained loop depth. Warnings must be visible in logs.

## 11.12 Rust Implementation Notes

Use typed structs with `serde`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognexisConfig {
    pub schema_version: u32,
    pub run: RunConfig,
    pub tokenizer: TokenizerConfig,
    pub model: ModelConfig,
    pub training: Option<TrainingConfig>,
    pub inference: Option<InferenceConfig>,
    pub evaluation: Option<EvaluationConfig>,
    pub safety: Option<SafetyConfig>,
    pub logging: Option<LoggingConfig>,
}
```

Implement a `validate()` method for every section and one top-level validation pass for cross-section constraints. Avoid interpreting raw maps throughout the codebase; convert to typed structs at the boundary.

## 11.13 Validation

Configuration tests must cover:

- Successful load of a complete valid config.
- Rejection of incompatible dimensions.
- Rejection of tokenizer/model vocab mismatch.
- Default resolution and resolved-config serialization.
- Command-line override application.
- Checkpoint config compatibility checks.
- Deprecation warnings for older schema versions.
