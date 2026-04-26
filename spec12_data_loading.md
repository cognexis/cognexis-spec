# 12. Data Loading

## 12.1 Purpose

The data loader streams tokenized or raw text examples into the training loop with high throughput and reproducible ordering. It must support large corpora, distributed workers, document packing, loss masks, loop curriculum metadata, and validation splits.

For Cognexis, data loading is not just I/O. It also prepares packed sequences whose boundaries affect causal masking, produces labels aligned for next-token prediction, and attaches loop-count or scheduler-training metadata used by recurrent training.

## 12.2 Supported Dataset Formats

The loader should support at least:

- JSONL records containing raw text or structured instruction examples.
- Pre-tokenized binary shards for high-throughput pretraining.
- Parquet or Arrow files for large metadata-rich corpora.
- Manifest files listing shard paths, sizes, checksums, and sampling weights.

A shard manifest entry should include:

```json
{
  "path": "shards/train_000123.bin",
  "format": "token_ids_u32",
  "num_documents": 100000,
  "num_tokens": 250000000,
  "checksum": "sha256:...",
  "weight": 1.0,
  "domain": "web"
}
```

The manifest is the unit of reproducibility. Training logs must record the manifest checksum.

## 12.3 Raw Text Processing

When loading raw text, the loader applies:

1. Record parsing.
2. Optional quality filters.
3. Prompt/chat formatting for supervised data.
4. Tokenizer encoding.
5. EOD/EOS insertion.
6. Packing into fixed-length training blocks.

Raw text loading is simpler operationally but can bottleneck training because tokenization is CPU-intensive. Large pretraining jobs should prefer pre-tokenized shards after validating tokenizer consistency.

## 12.4 Document Packing

Packing concatenates tokenized documents into fixed-length blocks:

```text
[doc_a tokens, EOD, doc_b tokens, EOD, doc_c prefix]
```

This reduces padding waste and improves accelerator utilization. The loader must also produce:

- `input_ids`: block except the final token.
- `target_ids`: block shifted one position to the left.
- `loss_mask`: 1 for tokens contributing to loss, 0 otherwise.
- Optional `document_ids` or `segment_ids` for boundary-aware attention.

If document-boundary attention is disabled, tokens after an EOD may attend to tokens before the EOD as normal packed-language-model training. If enabled, the attention mask must block cross-document attention.

## 12.5 Instruction Data

Instruction tuning examples are structured records:

```json
{
  "system": "You are a helpful assistant.",
  "messages": [
    {"role": "user", "content": "Explain ..."},
    {"role": "assistant", "content": "The answer is ..."}
  ],
  "metadata": {"source": "sft"}
}
```

The loader must render messages with the configured chat template and build a loss mask that trains only on assistant tokens unless the experiment explicitly chooses otherwise. User and system tokens should condition the model but not contribute to supervised response loss.

## 12.6 Batch Construction

A training batch contains:

```text
input_ids:      [batch, seq_len]
target_ids:     [batch, seq_len]
loss_mask:      [batch, seq_len]
attention_mask: [batch, seq_len] or structured mask metadata
position_ids:   [batch, seq_len]
loop_metadata:  per-batch or per-example loop settings
```

`seq_len` is usually `sequence_length - 1` when targets are shifted from a `sequence_length` token block. Implementations may store full blocks and perform shifting in the model loss path, but the behavior must be consistent.

## 12.7 Loop Metadata

For recurrent-depth training, each batch may carry:

- Fixed sampled loop count.
- Maximum loop count for adaptive training.
- Scheduler supervision targets.
- Whether intermediate states should be retained.
- Per-example task difficulty or curriculum tags.

The curriculum module may generate this metadata after data loading, but it must be attached before the forward pass. Distributed replicas must agree on loop metadata when gradients are synchronized across the same global step.

## 12.8 Shuffling

Shuffling must balance randomness and reproducibility. Recommended strategy:

- Shuffle shard order each epoch with a seeded RNG.
- Shuffle records within a buffer for streaming datasets.
- Use deterministic rank-based partitioning in distributed jobs.
- Record epoch, shard order seed, and consumed token offset in checkpoints.

For massive streaming corpora, exact global shuffling may be infeasible. In that case, use sufficiently large shuffle buffers and document the approximation.

## 12.9 Distributed Loading

In distributed training, each data-parallel rank must receive a distinct sample stream. Partitioning may be by shard, by record index, or by token block. Requirements:

- No systematic duplicate batches across ranks unless intentionally using data repetition.
- Same global batch shape across ranks.
- Consistent loop curriculum metadata across ranks when required.
- Checkpointable loader state per rank.
- Graceful handling of short or exhausted shards.

Validation datasets should be partitioned deterministically and evaluated without random shuffling.

## 12.10 Prefetching and Backpressure

The data loader should prefetch batches asynchronously:

```text
disk/network -> parse/tokenize -> pack -> host batch -> device transfer
```

Prefetch queues must be bounded to avoid unbounded memory growth. If the trainer falls behind, the loader should apply backpressure rather than reading unlimited data. Metrics should expose queue depth, load latency, tokenization latency, and skipped records.

## 12.11 Corruption and Error Policy

Corrupt records are expected in large corpora. The loader should support configurable policies:

- `fail`: stop training on first corrupt record.
- `skip`: skip corrupt records and log counts.
- `quarantine`: write corrupt record metadata to a side file.

For pretraining, `skip` with structured logging is usually acceptable. For evaluation, corruption should usually be fatal because it invalidates metrics.

## 12.12 Rust Implementation Notes

Recommended module structure:

```text
src/training/data/
  manifest.rs
  shard.rs
  tokenizer_worker.rs
  packer.rs
  batch.rs
  distributed.rs
```

Use iterators or async streams to compose loading stages. Keep batch tensors in contiguous memory before device transfer. Use memory mapping for binary token shards when feasible. Avoid loading entire large shards into memory.

## 12.13 Validation

Data-loading tests must cover:

- JSONL and pre-tokenized shard loading.
- Packing and EOD insertion.
- Input/target shift correctness.
- Loss masks for pretraining and instruction tuning.
- Deterministic shuffling with fixed seed.
- Distributed rank partitioning without overlap.
- Loader checkpoint and resume.
- Corrupt-record handling policies.
- Throughput metrics emission.
