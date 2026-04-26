# Cognexis Specification

Cognexis is a decoder-only, compute-adaptive recurrent-depth language model architecture. It separates stored parameter count from effective reasoning depth by reusing a shared recurrent transformer core for a configurable number of inference loops. The same checkpoint can therefore run at different quality, latency, and compute budgets.

This repository contains the engineering specification for Cognexis. The `spec*.md` files are the implementation contract; the white paper is a higher-level narrative document.

## Scope

The specification covers:

- Tokenization, embeddings, attention, feed-forward networks, transformer blocks, Prelude, Recurrent Core, Coda, and LM head.
- Configuration schemas, checkpoint compatibility, training data loading, loop curricula, and distributed training.
- Prefill/decode execution, KV cache semantics, recurrent stability controls, and adaptive loop scheduling.
- Value-head scheduling, token-wise loop allocation, evaluation metrics, loop scaling experiments, ablation studies, instruction tuning, safety, monitoring, and implementation structure.

The documents are written for a Rust implementation, but the architectural requirements are backend-independent. A conforming implementation may use any tensor runtime or GPU backend if it preserves the specified behavior, shapes, loop semantics, configuration fields, and evaluation outputs.

## Core Idea

Cognexis uses a three-stage model body:

```text
tokens
  -> embeddings
  -> Prelude
  -> Recurrent Core repeated N times
  -> Coda
  -> LM head
  -> logits
```

For a fixed loop count, effective depth is:

```text
effective_depth = num_prelude_blocks + loop_count * num_recurrent_blocks + num_coda_blocks
```

The baseline architecture uses exactly one shared recurrent block. Loop count may be fixed, sequence-adaptive, or token-wise adaptive, but every mode must enforce configured minimum loops, maximum loops, and request-level compute budgets.

## Reading Order

Start with [spec01_overview.md](spec01_overview.md). Then read the model component specs, followed by configuration/training, inference/scheduling, and evaluation/operations.

| File | Topic |
| --- | --- |
| [spec01_overview.md](spec01_overview.md) | System purpose, architecture, invariants, and document map |
| [spec02_tokenizer.md](spec02_tokenizer.md) | Tokenizer contract, vocabulary, special tokens, chat formatting, and validation |
| [spec03_embedding.md](spec03_embedding.md) | Token embeddings, RoPE position handling, initialization, and weight tying |
| [spec04_attention.md](spec04_attention.md) | Causal attention, GQA/MQA, RoPE, masks, and KV cache rules |
| [spec05_feedforward.md](spec05_feedforward.md) | FFN/SwiGLU architecture, dimensions, dropout, precision, and validation |
| [spec06_transformer_block.md](spec06_transformer_block.md) | Pre-norm block composition, residual scaling, gating hooks, and block contexts |
| [spec07_prelude.md](spec07_prelude.md) | Initial transformer stack before recurrence |
| [spec08_recurrent_core.md](spec08_recurrent_core.md) | Shared recurrent block, loop execution, scheduler interaction, and stability |
| [spec09_coda.md](spec09_coda.md) | Final transformer stack after recurrence |
| [spec10_lm_head.md](spec10_lm_head.md) | Final normalization, vocabulary logits, weight tying, and loss behavior |
| [spec11_config.md](spec11_config.md) | YAML/JSON configuration schema and validation rules |
| [spec12_data_loading.md](spec12_data_loading.md) | Dataset formats, packing, shuffling, batching, and distributed loading |
| [spec13_curriculum.md](spec13_curriculum.md) | Training-time loop curriculum and recurrent depth sampling |
| [spec14_distributed_training.md](spec14_distributed_training.md) | Parallelism, activation memory, precision, checkpointing, and fault tolerance |
| [spec15_stability_normalization.md](spec15_stability_normalization.md) | Normalization, residual controls, gating, spectral monitoring, and overthinking |
| [spec16_prefill_decode.md](spec16_prefill_decode.md) | Prefill/decode phases, cache ownership, streaming, budgets, and sampling |
| [spec17_scheduler_design.md](spec17_scheduler_design.md) | Adaptive loop scheduler inputs, policies, budgets, and diagnostics |
| [spec18_tokenwise_scheduling.md](spec18_tokenwise_scheduling.md) | Per-token halting, active masks, mixed-depth states, and sparse execution |
| [spec19_value_head.md](spec19_value_head.md) | Learned marginal-gain prediction, calibration, and confidence modeling |
| [spec20_evaluation_metrics.md](spec20_evaluation_metrics.md) | Quality, compute, DEI, LSP, OT, DGR, and scheduler metrics |
| [spec21_loop_scaling.md](spec21_loop_scaling.md) | Loop-depth experiments and quality-versus-compute analysis |
| [spec22_ablation.md](spec22_ablation.md) | Component ablations, sensitivity sweeps, and experiment design |
| [spec23_instruction_tuning.md](spec23_instruction_tuning.md) | SFT, code generation, depth controls, tool formats, and safety filtering |
| [spec24_safety_monitoring.md](spec24_safety_monitoring.md) | Runtime safety, budget enforcement, logs, metrics, tracing, and red teaming |
| [spec25_glossary.md](spec25_glossary.md) | Shared terminology, metric names, and halt reasons |
| [spec26_implementation_outline.md](spec26_implementation_outline.md) | Recommended Rust project structure, APIs, CLI, build, tests, and deployment |
| [spec27_limitations_future.md](spec27_limitations_future.md) | Known limitations and research roadmap |
| [spec28_conclusion.md](spec28_conclusion.md) | Summary, invariants, and recommended implementation path |

## Required Invariants

A Cognexis implementation must preserve these invariants:

- Tokenization is identical between training, evaluation, and inference for a given tokenizer artifact.
- Attention is causal in every decoder path.
- The recurrent core uses shared parameters across loop iterations.
- The scheduler respects `min_loops`, `max_loops`, and compute budgets.
- Evaluation reports quality and compute together.
- Checkpoints include resolved configuration and tokenizer metadata.
- Safety policies remain active regardless of loop mode.

## Recommended Implementation Path

1. Build a small reference implementation with fixed loops.
2. Validate tokenizer, masks, attention, FFN, transformer block, and LM loss.
3. Add Prelude, shared Recurrent Core, Coda, and final norm.
4. Implement prefill/decode and prove cache equivalence on small examples.
5. Add loop curriculum training and loop scaling evaluation.
6. Train and calibrate the value head.
7. Enable conservative adaptive scheduling.
8. Add token-wise scheduling and optimized distributed execution after correctness is established.

## Repository Contents

```text
README.md
cognexis_white_paper.md
white_paper_references.md
spec01_overview.md
...
spec28_conclusion.md
LICENSE
```

The `spec*.md` files are intended to be read as the normative technical specification. The white paper and references provide broader context.
